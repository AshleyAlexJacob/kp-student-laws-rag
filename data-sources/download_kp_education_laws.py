"""
Download student and education laws from the KP Code (kpcode.kp.gov.pk)
for use as a RAG corpus. Standard library only: run with `python download_kp_education_laws.py`.

Output:
  kp_education_laws/pdfs/<id>_<slug>.pdf   the official PDFs
  kp_education_laws/manifest.json          metadata per law (use it for citations and filtering)
"""
import json, re, ssl, time, urllib.request
from pathlib import Path

BASE = "https://kpcode.kp.gov.pk/homepage/lawDetails/"
OUT = Path("kp_education_laws")
DELAY_SECONDS = 2  # be polite to a government server

# (law_id, title, level, year)
LAWS = [
    # Higher education
    (879,  "Khyber Pakhtunkhwa Universities Act", "higher", 2012),
    (1214, "KP Educational Testing and Evaluation Agency (ETEA) Ordinance", "higher", 2001),
    (668,  "KP Higher Education Scholarship Endowment Fund Act", "higher", 2014),
    (1237, "Higher Education Research Endowment Fund Act", "higher", 2004),
    (502,  "West Pakistan Private Colleges (Management and Control) Ordinance", "higher", 1970),
    (1225, "Institute of Management Sciences Ordinance", "higher", 2002),
    (33,   "GIK Institute of Engineering Sciences and Technology Act", "higher", 1994),
    (1256, "Abasyn University Act", "higher", 2009),
    # School education
    (1325, "Free Compulsory Primary and Secondary Education Act", "school", 2017),
    (1359, "KP Private Schools Regulatory Authority Act", "school", 2017),
    (1208, "Registration and Functioning of Private Educational Institutions Ordinance", "school", 2001),
    (1415, "KP School Bags (Limitation of Weight) Act", "school", 2020),
    (1185, "Conduct of Examinations (Eradication of Malpractices) Act", "school", 1995),
    (23,   "Boards of Intermediate and Secondary Education Act", "school", 1990),
    (1273, "Supervision of Curricula, Textbooks and Maintenance of Standards of Education Act", "school", 2011),
    (1386, "KP Educational Monitoring Authority Act", "school", 2019),
    (504,  "KP Text Book Board Ordinance", "school", 1971),
    (503,  "West Pakistan Publication of Textbooks (Regulation and Control) Ordinance", "school", 1970),
    (1228, "Elementary and Secondary Education Foundation Ordinance", "school", 2002),
    (1180, "Education Foundation Act", "school", 1992),
    (1280, "Promotion of Regional Languages Authority Act", "school", 2012),
    (524,  "KP Government Education and Training Institutions Ordinance", "school", 1971),
    (532,  "Reformatory Schools Act", "school", 1897),
]

# Some government sites have certificate problems; fall back to an unverified
# context only if the verified request fails.
CTX_STRICT = ssl.create_default_context()
CTX_LOOSE = ssl._create_unverified_context()
HEADERS = {"User-Agent": "Mozilla/5.0 (educational RAG corpus builder)"}


def fetch(url, retries=3):
    for attempt in range(1, retries + 1):
        for ctx in (CTX_STRICT, CTX_LOOSE):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                    return r.read()
            except ssl.SSLError:
                continue  # try the loose context
            except Exception as e:
                err = e
                break
        print(f"    retry {attempt}/{retries} for {url}")
        time.sleep(DELAY_SECONDS * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {err}")


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:70]


def main():
    pdf_dir = OUT / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for law_id, title, level, year in LAWS:
        page_url = f"{BASE}{law_id}"
        target = pdf_dir / f"{law_id}_{slugify(title)}.pdf"
        print(f"[{law_id}] {title}")
        entry = {"id": law_id, "title": title, "level": level, "year": year,
                 "source_page": page_url, "file": str(target), "pdf_url": None, "status": None}
        try:
            html = fetch(page_url).decode("utf-8", errors="ignore")
            m = re.search(r'href="(https?://kpcode\.kp\.gov\.pk/uploads/[^"]+\.pdf)"', html, re.I)
            if not m:
                entry["status"] = "no_pdf_link"
                print("    no PDF link found on page")
            else:
                entry["pdf_url"] = m.group(1)
                if target.exists() and target.stat().st_size > 0:
                    entry["status"] = "already_downloaded"
                else:
                    data = fetch(entry["pdf_url"])
                    if not data.startswith(b"%PDF"):
                        entry["status"] = "not_a_pdf"
                    else:
                        target.write_bytes(data)
                        entry["status"] = "ok"
                        print(f"    saved {len(data)//1024} KB")
        except Exception as e:
            entry["status"] = f"error: {e}"
            print(f"    {entry['status']}")
        manifest.append(entry)
        time.sleep(DELAY_SECONDS)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(e["status"] in ("ok", "already_downloaded") for e in manifest)
    print(f"\nDone: {ok}/{len(manifest)} PDFs in {pdf_dir}. Manifest: {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
