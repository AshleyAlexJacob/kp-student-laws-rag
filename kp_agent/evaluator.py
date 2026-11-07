"""Evaluation module for KP Education Laws Agent.

Implements the RagEvaluator class integrating the Ragas evaluation framework
assessing RAG pipeline quality across faithfulness, answer relevance, context recall,
and statutory correctness using the canonical benchmark test set (data/test_set.json).
"""

import argparse
import csv
import json
import math
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Compatibility shim: ragas 0.3.1 contains an unconditioned import of
# langchain_community.chat_models.vertexai, which was removed in langchain-community >= 0.3.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _v_chat_mod = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        """Compatibility dummy for removed langchain_community ChatVertexAI."""
        pass

    _v_chat_mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _v_chat_mod

from ragas import RunConfig, evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)

from kp_agent.agent import LawAgent
from kp_agent.config import EvaluationConfig, Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory

logger = get_logger(__name__)


def _clean_score(val: Any) -> float:
    """Safely convert metric score to float, mapping NaN or None to 0.0.

    Args:
        val: Metric score value from Ragas result.

    Returns:
        float: Normalized float score between 0.0 and 1.0.
    """
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else round(f, 4)
    except (ValueError, TypeError):
        return 0.0


@dataclass
class EvalResult:
    """Evaluation score breakdown for a single test set question."""

    question_id: str
    question: str
    question_type: str
    ground_truth: str
    agent_answer: str
    sources_retrieved: int
    faithfulness: float
    relevance: float
    context_recall: float
    correctness: float
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class RagEvaluator:
    """Evaluates agent responses against ground truth benchmarks using Ragas."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        agent: Optional[LawAgent] = None,
        eval_config: Optional[EvaluationConfig] = None,
    ) -> None:
        """Initialize the RAG evaluator with agent, judge LLM, and embeddings model.

        Args:
            settings: Application configuration settings. Defaults to singleton.
            agent: LawAgent instance under evaluation.
            eval_config: Evaluation configuration from evaluation.yaml.
        """
        self.settings = settings or get_settings()
        self.agent = agent or LawAgent(settings=self.settings, enable_cache=False)
        self.eval_config = eval_config or self.settings.evaluation

        # Configure evaluation judge model and embeddings based on evaluation.yaml
        models_cfg = self.eval_config.models
        judge_provider = models_cfg.provider or self.settings.provider
        judge_model_name = models_cfg.judge_model or self.settings.chat_model
        embed_model_name = models_cfg.embedding_model or self.settings.embedding_model

        logger.info(
            "Initializing Ragas evaluation models: Provider=%s, Judge=%s, Embeddings=%s",
            judge_provider,
            judge_model_name,
            embed_model_name,
        )

        self.judge_model = ModelFactory.get_chat_model(
            self.settings,
            temperature=models_cfg.temperature,
            model_name=judge_model_name,
            provider=judge_provider,
        )
        self.embeddings_model = ModelFactory.get_embeddings(
            self.settings,
            model_name=embed_model_name,
            provider=judge_provider,
        )

    def load_test_set(self, test_set_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Load evaluation test set questions and ground truths.

        Args:
            test_set_path: Path to test_set.json. Defaults to configured path.

        Returns:
            List[Dict[str, Any]]: Parsed test cases.

        Raises:
            FileNotFoundError: If the test set JSON file does not exist.
        """
        raw_path = test_set_path or Path(self.eval_config.benchmark.test_set_path)
        path = raw_path if raw_path.is_absolute() else (self.settings.base_dir / raw_path)

        if not path.exists():
            logger.error("Test set file not found at: %s", path)
            raise FileNotFoundError(f"Test set file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Loaded %d test set questions from %s", len(data), path.name)
        return data

    def _build_metrics(self) -> List[Any]:
        """Construct list of active Ragas metrics based on evaluation.yaml feature flags.

        Returns:
            List[Any]: Configured Ragas metric instances.
        """
        flags = self.eval_config.feature_flags
        active_metrics: List[Any] = []

        if flags.enable_faithfulness:
            active_metrics.append(faithfulness)
        if flags.enable_answer_relevance:
            active_metrics.append(answer_relevancy)
        if flags.enable_context_recall:
            active_metrics.append(context_recall)
        if flags.enable_answer_correctness:
            active_metrics.append(answer_correctness)
        if flags.enable_context_precision:
            active_metrics.append(context_precision)
        if flags.enable_semantic_similarity:
            active_metrics.append(answer_similarity)

        logger.info("Configured %d Ragas metrics for evaluation run.", len(active_metrics))
        return active_metrics

    def evaluate_all(
        self,
        limit: Optional[int] = None,
        question_id: Optional[str] = None,
    ) -> List[EvalResult]:
        """Run complete Ragas evaluation over the benchmark test set.

        Args:
            limit: Optional maximum number of test questions to evaluate.
            question_id: Optional single question ID to evaluate.

        Returns:
            List[EvalResult]: List of evaluated sample results.
        """
        test_set = self.load_test_set()

        # Filter by single ID if requested
        if question_id:
            test_set = [item for item in test_set if item.get("id") == question_id]
            if not test_set:
                logger.error("No question found matching ID: %s", question_id)
                return []

        # Apply sample limit (explicit argument takes priority over YAML default)
        effective_limit = limit if limit is not None else self.eval_config.benchmark.default_sample_limit
        if effective_limit is not None and effective_limit > 0:
            test_set = test_set[:effective_limit]

        total = len(test_set)
        logger.info("Beginning RAG evaluation on %d test questions with Ragas...", total)

        samples: List[SingleTurnSample] = []
        raw_meta: List[Dict[str, Any]] = []

        # Step 1: Collect agent responses and retrieved context chunks
        for idx, item in enumerate(test_set, start=1):
            q_id = item.get("id", f"q-{idx}")
            question = item["question"]
            gt = item["ground_truth"]
            q_type = item.get("question_type", "factual")

            logger.info("[%d/%d] Asking LawAgent (%s - %s): '%s'", idx, total, q_id, q_type, question[:60])

            # Ask agent with dedicated thread_id for isolation
            agent_res = self.agent.ask(query=question, thread_id=f"eval_{q_id}", use_cache=False)
            answer = agent_res.get("answer", "")
            sources = agent_res.get("sources", [])

            # Extract retrieved statutory text snippets
            snippets = [s.get("snippet", "").strip() for s in sources if s.get("snippet")]
            # Guarantee at least one context string so Ragas does not fail on empty contexts
            contexts = snippets if snippets else ["No statutory context retrieved."]

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                reference=gt,
                retrieved_contexts=contexts,
            )
            samples.append(sample)
            raw_meta.append({
                "id": q_id,
                "question": question,
                "question_type": q_type,
                "ground_truth": gt,
                "agent_answer": answer,
                "sources_retrieved": len(sources),
            })

        # Step 2: Build EvaluationDataset and configure Ragas RunConfig
        eval_dataset = EvaluationDataset(samples=samples)
        metrics = self._build_metrics()

        ragas_cfg = self.eval_config.ragas_config
        run_config = RunConfig(
            timeout=ragas_cfg.timeout,
            max_retries=ragas_cfg.max_retries,
            max_workers=ragas_cfg.max_workers,
        )

        logger.info(
            "Executing ragas.evaluate (workers=%d, timeout=%ds, retries=%d)...",
            ragas_cfg.max_workers,
            ragas_cfg.timeout,
            ragas_cfg.max_retries,
        )

        # Step 3: Execute Ragas evaluation
        eval_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.judge_model,
            embeddings=self.embeddings_model,
            run_config=run_config,
            raise_exceptions=ragas_cfg.raise_exceptions,
            show_progress=ragas_cfg.show_progress,
            batch_size=ragas_cfg.batch_size,
        )

        # Step 4: Map Ragas result scores to EvalResult instances
        scores_list = eval_result.scores
        results: List[EvalResult] = []

        for idx, meta in enumerate(raw_meta):
            score_dict = scores_list[idx] if idx < len(scores_list) else {}
            faith = _clean_score(score_dict.get("faithfulness"))
            rel = _clean_score(score_dict.get("answer_relevancy"))
            rec = _clean_score(score_dict.get("context_recall"))
            corr = _clean_score(score_dict.get("answer_correctness"))

            # Extract any additional enabled metric scores
            extra = {
                k: _clean_score(v)
                for k, v in score_dict.items()
                if k not in {"faithfulness", "answer_relevancy", "context_recall", "answer_correctness"}
            }

            res = EvalResult(
                question_id=meta["id"],
                question=meta["question"],
                question_type=meta["question_type"],
                ground_truth=meta["ground_truth"],
                agent_answer=meta["agent_answer"],
                sources_retrieved=meta["sources_retrieved"],
                faithfulness=faith,
                relevance=rel,
                context_recall=rec,
                correctness=corr,
                extra_metrics=extra,
            )
            results.append(res)

        return results

    def print_summary(self, results: List[EvalResult]) -> None:
        """Print structured tabular and aggregate evaluation scores to console.

        Args:
            results: List of completed evaluation results.
        """
        if not results:
            print("No evaluation results to summarize.")
            return

        print("\n" + "=" * 90)
        print(f"KHYBER PAKHTUNKHWA EDUCATION LAWS RAG AGENT - RAGAS EVALUATION ({len(results)} SAMPLES)")
        print("=" * 90)
        print(f"{'ID':<8} | {'Type':<14} | {'Faith.':<6} | {'Relev.':<6} | {'Recall':<6} | {'Corr.':<6} | {'Sources':<7}")
        print("-" * 90)

        for r in results:
            print(
                f"{r.question_id:<8} | {r.question_type:<14} | {r.faithfulness:<6.2f} | "
                f"{r.relevance:<6.2f} | {r.context_recall:<6.2f} | {r.correctness:<6.2f} | {r.sources_retrieved:<7}"
            )

        print("-" * 90)

        # Compute aggregate averages
        n = len(results)
        avg_faith = sum(r.faithfulness for r in results) / n
        avg_rel = sum(r.relevance for r in results) / n
        avg_rec = sum(r.context_recall for r in results) / n
        avg_corr = sum(r.correctness for r in results) / n

        print(f"{'OVERALL MEAN':<25} | {avg_faith:<6.2f} | {avg_rel:<6.2f} | {avg_rec:<6.2f} | {avg_corr:<6.2f} |")
        print("=" * 90)

        # Breakdown by question type
        by_type: Dict[str, List[EvalResult]] = {}
        for r in results:
            by_type.setdefault(r.question_type, []).append(r)

        print("\nBREAKDOWN BY QUESTION TYPE:")
        print(f"{'Type':<16} | {'Count':<5} | {'Faithfulness':<12} | {'Relevance':<10} | {'Recall':<8} | {'Correctness':<11}")
        print("-" * 75)
        for t_name, t_list in by_type.items():
            t_n = len(t_list)
            print(
                f"{t_name:<16} | {t_n:<5} | "
                f"{sum(x.faithfulness for x in t_list) / t_n:<12.2f} | "
                f"{sum(x.relevance for x in t_list) / t_n:<10.2f} | "
                f"{sum(x.context_recall for x in t_list) / t_n:<8.2f} | "
                f"{sum(x.correctness for x in t_list) / t_n:<11.2f}"
            )
        print("=" * 75 + "\n")

    def save_results(
        self,
        results: List[EvalResult],
        output_json_path: Optional[Path] = None,
        output_csv_path: Optional[Path] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """Save detailed evaluation results to JSON and CSV files.

        Args:
            results: List of evaluation result records.
            output_json_path: Target JSON filepath.
            output_csv_path: Target CSV filepath.

        Returns:
            Tuple[Path, Optional[Path]]: Paths to written JSON and CSV files.
        """
        raw_json = output_json_path or Path(self.eval_config.benchmark.output_json)
        json_file = raw_json if raw_json.is_absolute() else (self.settings.base_dir / raw_json)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON report
        payload = [asdict(r) for r in results]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved evaluation JSON report to %s", json_file)

        # Write CSV report if requested or configured
        csv_file: Optional[Path] = None
        if self.eval_config.benchmark.save_detailed_reports:
            raw_csv = output_csv_path or Path(self.eval_config.benchmark.output_csv)
            csv_file = raw_csv if raw_csv.is_absolute() else (self.settings.base_dir / raw_csv)
            csv_file.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = [
                "question_id",
                "question_type",
                "faithfulness",
                "relevance",
                "context_recall",
                "correctness",
                "sources_retrieved",
                "question",
                "ground_truth",
                "agent_answer",
            ]
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    writer.writerow(asdict(r))
            logger.info("Saved evaluation CSV report to %s", csv_file)

        return json_file, csv_file


def main() -> None:
    """CLI entrypoint for running RAG evaluation with Ragas."""
    parser = argparse.ArgumentParser(description="Evaluate KP Education Laws Agent against test set with Ragas.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of test questions to evaluate.")
    parser.add_argument("--id", type=str, default=None, help="Evaluate a single question by its ID (e.g. kp-01).")
    parser.add_argument("--config", type=str, default=None, help="Custom path to evaluation.yaml.")
    args = parser.parse_args()

    settings = get_settings()
    eval_config = None
    if args.config:
        custom_cfg_path = Path(args.config)
        if custom_cfg_path.exists():
            eval_config = EvaluationConfig.load_from_yaml(custom_cfg_path)
        else:
            logger.warning("Custom config path %s not found, using default.", args.config)

    evaluator = RagEvaluator(settings=settings, eval_config=eval_config)
    results = evaluator.evaluate_all(limit=args.limit, question_id=args.id)
    evaluator.print_summary(results)
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
