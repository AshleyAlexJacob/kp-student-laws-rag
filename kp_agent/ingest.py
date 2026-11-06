"""Legal document ingestion and hierarchical semantic chunking pipeline.

Parses KP education law PDFs into document-aware semantic chunks preserving
statutory hierarchy (Preamble, Sections, Subsections, Clauses, Definitions,
Schedules and Tables) across page boundaries, embeds with OpenAI text-embedding-3-large,
and indexes into a local or Docker Qdrant vector database.
"""

import argparse
import csv
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from kp_agent.config import Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory

logger = get_logger(__name__)


@dataclass
class LegalChunk:
    """Represents a hierarchical semantic chunk extracted from a legal document."""

    text: str  # Original substantive statutory text
    context_text: str  # Text prepended with document and hierarchical context for embedding
    metadata: Dict[str, Any]  # Statutory and citation metadata
    chunk_id: str  # Unique deterministic identifier


class LegalDocParser:
    """Parser that decomposes legal PDF text into structural semantic units."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the legal document parser.

        Args:
            settings: Application settings containing directory paths.
        """
        self.settings = settings or get_settings()

    def is_noise_line(self, line: str) -> bool:
        """Determine if a line is a header, footer, or isolated page number.

        Args:
            line: The text line to evaluate.

        Returns:
            bool: True if the line is noise and should be discarded.
        """
        clean = line.strip()
        if not clean:
            return True

        # Isolated page numbers: e.g. "1", "- 2 -", "Page 3"
        if re.match(r"^(?:Page\s+)?-?\s*\d+\s*-?$", clean, re.IGNORECASE):
            return True

        # Running gazette or government publication headers
        if re.search(
            r"(?:KHYBER\s+PAKHTUNKHWA\s+ACT\s+NO|EXTRAORDINARY|GOVERNMENT\s+GAZETTE|REGISTERED\s+No)",
            clean,
            re.IGNORECASE,
        ):
            return True

        return False

    def extract_clean_lines(self, pdf_path: Path) -> Tuple[List[Tuple[str, int]], List[int]]:
        """Extract lines and associate them with their 1-based page numbers.

        Args:
            pdf_path: Path to the target PDF document.

        Returns:
            Tuple[List[Tuple[str, int]], List[int]]: List of (clean_line, page_number)
            and a list of page numbers flagged as scanned (< 50 chars).
        """
        reader = PdfReader(str(pdf_path))
        line_records: List[Tuple[str, int]] = []
        scanned_pages: List[int] = []

        for idx, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            stripped = raw_text.strip()

            # Rule: pages with fewer than 50 characters are flagged as scanned
            if len(stripped) < 50:
                logger.warning(
                    "Scanned page detected (fewer than 50 characters): %s (Page %d, chars: %d). Flagged for OCR.",
                    pdf_path.name,
                    idx,
                    len(stripped),
                )
                scanned_pages.append(idx)
                continue

            for raw_line in raw_text.splitlines():
                if not self.is_noise_line(raw_line):
                    line_records.append((raw_line.strip(), idx))

        return line_records, scanned_pages

    def build_markdown_table(self, lines: List[str]) -> str:
        """Format tabular lines from schedules into a structured Markdown table.

        Args:
            lines: Raw text lines identified as schedule table content.

        Returns:
            str: Markdown formatted table representation.
        """
        table_rows: List[List[str]] = []
        for line in lines:
            # Match multi-column data split by tabs, multiple spaces, or pipes
            cols = [c.strip() for c in re.split(r"\t+|\s{2,}|\|", line) if c.strip()]
            if cols:
                table_rows.append(cols)

        if not table_rows:
            return "\n".join(lines)

        # Standardize max columns
        max_cols = max(len(row) for row in table_rows)
        md_lines = []

        # If header exists, format with separator
        header = table_rows[0]
        while len(header) < max_cols:
            header.append("")
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        for row in table_rows[1:]:
            while len(row) < max_cols:
                row.append("")
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    def parse_document(self, pdf_path: Path, doc_meta: Dict[str, Any]) -> List[LegalChunk]:
        """Parse a law PDF into hierarchical semantic chunks.

        Args:
            pdf_path: Absolute or relative path to the PDF.
            doc_meta: Document metadata dictionary from metadata.csv.

        Returns:
            List[LegalChunk]: Complete list of structured semantic chunks.
        """
        line_records, _ = self.extract_clean_lines(pdf_path)
        if not line_records:
            logger.warning("No extractable text found in %s", pdf_path.name)
            return []

        doc_title = doc_meta.get("title", "")
        year = int(doc_meta.get("year", 0))
        level = doc_meta.get("level", "")
        source_url = doc_meta.get("source_url", "")
        file_name = doc_meta.get("file", pdf_path.name)

        doc_type = "Act" if "Act" in doc_title else "Ordinance" if "Ordinance" in doc_title else "Regulation"

        chunks: List[LegalChunk] = []

        # State tracking for document structure
        in_toc = False
        in_schedule = False
        current_schedule_name = ""
        schedule_lines: List[Tuple[str, int]] = []

        preamble_lines: List[Tuple[str, int]] = []
        in_preamble = False

        current_sec_num = ""
        current_sec_title = ""
        current_sec_lines: List[Tuple[str, int]] = []

        # Regular expressions for Pakistani law structures
        toc_pattern = re.compile(r"^(?:TABLE\s+OF\s+CONTENTS|ARRANGEMENT\s+OF\s+SECTIONS|CONTENTS)\b", re.IGNORECASE)
        preamble_pattern = re.compile(r"^(?:PREAMBLE|WHEREAS\b|AN\s+ACT\b)", re.IGNORECASE)
        section_pattern = re.compile(
            r"^(?:(?:Section|SEC\.)\s+)?(\d+[A-Z]?)\.\s+([A-Z][^\n.—]{2,80}?)(?:[.—]|\s*$)",
            re.IGNORECASE,
        )
        schedule_pattern = re.compile(
            r"^(?:THE\s+)?(SCHEDULE(?:\s*[-–]\s*[I|V|X\d]+)?|APPENDIX(?:\s*[-–]\s*[I|V|X\d]+)?)\b",
            re.IGNORECASE,
        )

        def flush_schedule() -> None:
            """Flush accumulated schedule content as a chunk."""
            if not schedule_lines:
                return
            sched_text_raw = [text for text, _ in schedule_lines]
            sched_formatted = self.build_markdown_table(sched_text_raw)
            p_start = schedule_lines[0][1]
            p_end = schedule_lines[-1][1]

            ctx_text = (
                f"[Document: {doc_title} ({year}) | {level.capitalize()} Education]\n"
                f"[Schedule: {current_schedule_name}]\n\n"
                f"{sched_formatted}"
            )

            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_{current_schedule_name}_{p_start}"))
            chunks.append(
                LegalChunk(
                    text=sched_formatted,
                    context_text=ctx_text,
                    metadata={
                        "document_title": doc_title,
                        "document_type": doc_type,
                        "title": doc_title,
                        "year": year,
                        "level": level,
                        "source_url": source_url,
                        "file": file_name,
                        "section_number": current_schedule_name,
                        "section_title": current_schedule_name,
                        "subsection_number": None,
                        "clause_number": None,
                        "schedule_name": current_schedule_name,
                        "parent_section": None,
                        "page_start": p_start,
                        "page_end": p_end,
                        "page": p_start,
                    },
                    chunk_id=chunk_id,
                )
            )

        def flush_preamble() -> None:
            """Flush accumulated preamble text as a chunk."""
            if not preamble_lines:
                return
            p_text = " ".join(t for t, _ in preamble_lines)
            p_start = preamble_lines[0][1]
            p_end = preamble_lines[-1][1]

            ctx_text = (
                f"[Document: {doc_title} ({year}) | {level.capitalize()} Education]\n"
                f"[Context: Preamble]\n\n"
                f"{p_text}"
            )
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_preamble_{p_start}"))
            chunks.append(
                LegalChunk(
                    text=p_text,
                    context_text=ctx_text,
                    metadata={
                        "document_title": doc_title,
                        "document_type": doc_type,
                        "title": doc_title,
                        "year": year,
                        "level": level,
                        "source_url": source_url,
                        "file": file_name,
                        "section_number": "Preamble",
                        "section_title": "Preamble",
                        "subsection_number": None,
                        "clause_number": None,
                        "schedule_name": None,
                        "parent_section": None,
                        "page_start": p_start,
                        "page_end": p_end,
                        "page": p_start,
                    },
                    chunk_id=chunk_id,
                )
            )

        def flush_section() -> None:
            """Process and emit chunks for the current section."""
            if not current_sec_lines:
                return

            p_start = current_sec_lines[0][1]
            p_end = current_sec_lines[-1][1]
            full_sec_text = "\n".join(t for t, _ in current_sec_lines)
            sec_label = f"Section {current_sec_num}"

            # Special Handling: Definitions Section (typically Section 2)
            if "definition" in current_sec_title.lower() or current_sec_num == "2":
                # Extract individual definition clauses: (a) "...", (b) "..."
                def_pattern = re.compile(
                    r"(?:^|\n)\s*(\([a-z\d]+\))\s*([\"'“‘][^\"'”’]+[\"'”’]\s*means\b.+?)(?=(?:\n\s*\([a-z\d]+\)|\Z))",
                    re.DOTALL | re.IGNORECASE,
                )
                definitions = list(def_pattern.finditer(full_sec_text))

                if definitions:
                    for d_match in definitions:
                        clause_id = d_match.group(1).strip()
                        def_body = d_match.group(2).strip()

                        ctx_text = (
                            f"[Document: {doc_title} ({year}) | {level.capitalize()} Education]\n"
                            f"[Section: {sec_label} - {current_sec_title} | Clause: {clause_id}]\n\n"
                            f"{clause_id} {def_body}"
                        )
                        chunk_id = str(
                            uuid.uuid5(
                                uuid.NAMESPACE_DNS,
                                f"{file_name}_{sec_label}_{clause_id}_{p_start}",
                            )
                        )
                        chunks.append(
                            LegalChunk(
                                text=f"{clause_id} {def_body}",
                                context_text=ctx_text,
                                metadata={
                                    "document_title": doc_title,
                                    "document_type": doc_type,
                                    "title": doc_title,
                                    "year": year,
                                    "level": level,
                                    "source_url": source_url,
                                    "file": file_name,
                                    "section_number": sec_label,
                                    "section_title": current_sec_title,
                                    "subsection_number": None,
                                    "clause_number": clause_id,
                                    "schedule_name": None,
                                    "parent_section": sec_label,
                                    "page_start": p_start,
                                    "page_end": p_end,
                                    "page": p_start,
                                },
                                chunk_id=chunk_id,
                            )
                        )
                    return

            # Splitting large sections (>1200 characters) into subsection-level chunks
            if len(full_sec_text) > 1200:
                subsec_pattern = re.compile(
                    r"(?:^|\n)\s*(\(\d+\))\s*(.+?)(?=(?:\n\s*\(\d+\)|\Z))",
                    re.DOTALL,
                )
                subsections = list(subsec_pattern.finditer(full_sec_text))
                if len(subsections) > 1:
                    for s_match in subsections:
                        sub_id = s_match.group(1).strip()
                        sub_body = s_match.group(2).strip()

                        ctx_text = (
                            f"[Document: {doc_title} ({year}) | {level.capitalize()} Education]\n"
                            f"[Section: {sec_label} - {current_sec_title} | Subsection: {sub_id}]\n\n"
                            f"{sub_id} {sub_body}"
                        )
                        chunk_id = str(
                            uuid.uuid5(
                                uuid.NAMESPACE_DNS,
                                f"{file_name}_{sec_label}_{sub_id}_{p_start}",
                            )
                        )
                        chunks.append(
                            LegalChunk(
                                text=f"{sub_id} {sub_body}",
                                context_text=ctx_text,
                                metadata={
                                    "document_title": doc_title,
                                    "document_type": doc_type,
                                    "title": doc_title,
                                    "year": year,
                                    "level": level,
                                    "source_url": source_url,
                                    "file": file_name,
                                    "section_number": sec_label,
                                    "section_title": current_sec_title,
                                    "subsection_number": sub_id,
                                    "clause_number": None,
                                    "schedule_name": None,
                                    "parent_section": sec_label,
                                    "page_start": p_start,
                                    "page_end": p_end,
                                    "page": p_start,
                                },
                                chunk_id=chunk_id,
                            )
                        )
                    return

            # Default: Entire section as primary semantic unit
            ctx_text = (
                f"[Document: {doc_title} ({year}) | {level.capitalize()} Education]\n"
                f"[Section: {sec_label} - {current_sec_title}]\n\n"
                f"{full_sec_text}"
            )
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_{sec_label}_{p_start}"))
            chunks.append(
                LegalChunk(
                    text=full_sec_text,
                    context_text=ctx_text,
                    metadata={
                        "document_title": doc_title,
                        "document_type": doc_type,
                        "title": doc_title,
                        "year": year,
                        "level": level,
                        "source_url": source_url,
                        "file": file_name,
                        "section_number": sec_label,
                        "section_title": current_sec_title,
                        "subsection_number": None,
                        "clause_number": None,
                        "schedule_name": None,
                        "parent_section": sec_label,
                        "page_start": p_start,
                        "page_end": p_end,
                        "page": p_start,
                    },
                    chunk_id=chunk_id,
                )
            )

        # Iterate through text lines and map structural sections
        for line, p_num in line_records:
            # Check Table of Contents boundary
            if toc_pattern.search(line):
                in_toc = True
                continue

            # Schedule match
            sched_match = schedule_pattern.match(line)
            if sched_match:
                in_toc = False
                in_preamble = False
                flush_section()
                flush_schedule()
                in_schedule = True
                current_schedule_name = sched_match.group(1).strip()
                schedule_lines = [(line, p_num)]
                continue

            if in_schedule:
                schedule_lines.append((line, p_num))
                continue

            # Section match
            sec_match = section_pattern.match(line)
            if sec_match:
                # If we were in TOC, the first section with long text ends the TOC
                if in_toc:
                    # If this line is just a brief TOC entry (e.g. "1. Short title..."), ignore
                    if len(line) < 60:
                        continue
                    in_toc = False

                in_preamble = False
                flush_section()
                flush_preamble()

                current_sec_num = sec_match.group(1).strip()
                current_sec_title = sec_match.group(2).strip()
                current_sec_lines = [(line, p_num)]
                continue

            # Preamble match
            if preamble_pattern.search(line) and not current_sec_num:
                in_toc = False
                in_preamble = True
                preamble_lines.append((line, p_num))
                continue

            if in_preamble:
                preamble_lines.append((line, p_num))
                continue

            if current_sec_num:
                current_sec_lines.append((line, p_num))
            elif not in_toc:
                # Leading metadata or preamble text before first numbered section
                preamble_lines.append((line, p_num))

        # Flush any trailing structures
        flush_preamble()
        flush_section()
        flush_schedule()

        return chunks


class LawIngestor:
    """Manages reading legal documents, generating embeddings, and indexing into Qdrant."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize LawIngestor with settings and Qdrant client.

        Args:
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.parser = LegalDocParser(self.settings)

        # Initialize Qdrant Client
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )

    def load_metadata(self) -> List[Dict[str, Any]]:
        """Load document metadata from data/metadata.csv.

        Returns:
            List[Dict[str, Any]]: List of metadata records.
        """
        meta_path = self.settings.metadata_file
        if not meta_path.exists():
            logger.error("Metadata file not found at: %s", meta_path)
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        records = []
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        logger.info("Loaded %d law metadata records from %s", len(records), meta_path)
        return records

    def ensure_collection(self, recreate: bool = False) -> bool:
        """Ensure the target Qdrant collection is created with correct dimensions.

        Args:
            recreate: If True, recreate the collection even if it already exists.

        Returns:
            bool: True if collection is ready for indexing, False if existing and skipping.
        """
        col_name = self.settings.qdrant_collection
        exists = self.client.collection_exists(collection_name=col_name)

        if exists and not recreate:
            logger.info("Collection '%s' already exists in Qdrant. Skipping re-indexing.", col_name)
            return False

        logger.info(
            "Creating collection '%s' with vector size=%d (Cosine distance)",
            col_name,
            self.settings.embedding_dimension,
        )
        self.client.recreate_collection(
            collection_name=col_name,
            vectors_config=VectorParams(
                size=self.settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

        # Create payload indexes for fast filtering
        for field, schema in [
            ("level", PayloadSchemaType.KEYWORD),
            ("year", PayloadSchemaType.INTEGER),
            ("section_number", PayloadSchemaType.KEYWORD),
            ("document_title", PayloadSchemaType.KEYWORD),
        ]:
            self.client.create_payload_index(
                collection_name=col_name,
                field_name=field,
                field_schema=schema,
            )

        logger.info("Created payload indexes on: level, year, section_number, document_title.")
        return True

    def ingest(self, recreate: bool = False, batch_size: int = 64) -> int:
        """Execute full ingestion of all 23 laws into Qdrant.

        Args:
            recreate: Whether to wipe and recreate the Qdrant collection.
            batch_size: Number of chunks per embedding & upsert batch.

        Returns:
            int: Total number of chunks indexed.
        """
        # Step 1: Check collection existence
        ready = self.ensure_collection(recreate=recreate)
        if not ready:
            return 0

        # Step 2: Parse all laws into semantic chunks
        metadata_list = self.load_metadata()
        all_chunks: List[LegalChunk] = []

        for meta in metadata_list:
            pdf_file = meta.get("file", "")
            pdf_path = self.settings.laws_dir / pdf_file

            if not pdf_path.exists():
                logger.warning("PDF file not found at %s. Skipping.", pdf_path)
                continue

            doc_chunks = self.parser.parse_document(pdf_path, meta)
            logger.info("Extracted %d chunks from %s", len(doc_chunks), pdf_file)
            all_chunks.extend(doc_chunks)

        total_chunks = len(all_chunks)
        logger.info("Total hierarchical chunks to index across 23 laws: %d", total_chunks)

        if total_chunks == 0:
            logger.warning("No chunks were extracted. Aborting ingestion.")
            return 0

        # Step 3: Embed and upload in batches
        embeddings_model = ModelFactory.get_embeddings(self.settings)

        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i : i + batch_size]
            batch_texts = [c.context_text for c in batch]

            logger.info(
                "Embedding batch %d-%d of %d with %s...",
                i + 1,
                min(i + batch_size, total_chunks),
                total_chunks,
                self.settings.embedding_model,
            )

            vectors = embeddings_model.embed_documents(batch_texts)

            points = [
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        **chunk.metadata,
                        "text": chunk.text,
                        "context_text": chunk.context_text,
                    },
                )
                for chunk, vector in zip(batch, vectors)
            ]

            self.client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=points,
            )

        logger.info("Successfully indexed %d chunks into collection '%s'.", total_chunks, self.settings.qdrant_collection)
        return total_chunks


def main() -> None:
    """CLI entry point for running ingestion."""
    parser = argparse.ArgumentParser(description="Ingest KP Education Laws into Qdrant vector database.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if the collection already exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding and vector upserts (default: 64).",
    )
    args = parser.parse_args()

    ingestor = LawIngestor()
    ingestor.ingest(recreate=args.force, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

