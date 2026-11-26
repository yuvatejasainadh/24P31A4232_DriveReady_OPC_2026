"""Task 5: Automated CSV & JSON Evaluation Framework for Day 3.

Evaluates the UniversalAgentWorkflow across seven deterministic dimensions:
  1. Intent Accuracy
  2. Entity Accuracy
  3. Tool Selection Accuracy
  4. Tool Execution Success
  5. Grounding Compliance
  6. Response Relevance (explicitly labeled deterministic_heuristic)
  7. Overall Pass Rate

Generates structured JSON reports and readable terminal summaries.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

DAY2_DIR = Path(__file__).resolve().parent.parent / "Day-2"
if str(DAY2_DIR) not in sys.path:
    sys.path.append(str(DAY2_DIR))

from agent_workflow import UniversalAgentWorkflow, WorkflowResult  # noqa: E402
from business_config import ConfigurationError, ValidationError  # noqa: E402
from chains import LCEL_BACKEND  # noqa: E402
from grounding import SafetyGuard  # noqa: E402


DEFAULT_CSV_PATH = Path(__file__).with_name("test.csv")
DEFAULT_JSON_CASES_PATH = Path(__file__).with_name("data") / "evaluation_cases.json"


@dataclass(frozen=True)
class EvaluationCase:
    """Normalized test case loaded from CSV or JSON."""

    test_id: str
    business_profile: str
    user_query: str
    expected_intent: str
    expected_tool: str | None
    expected_entity: dict[str, Any]
    expected_response_type: str
    prior_queries: list[str]
    must_include: list[str]


def parse_expected_entity(raw: Any) -> dict[str, Any]:
    """Parse expected entities from dict, JSON string, or 'k=v;k2=v2' CSV text."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValidationError(f"Unsupported expected_entity type: {type(raw).__name__}")

    cleaned = raw.strip()
    if not cleaned:
        return {}
    if cleaned.startswith("{"):
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed

    result: dict[str, Any] = {}
    for pair in cleaned.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", v):
            result[k] = float(v)
        else:
            result[k] = v
    return result


def load_evaluation_cases(dataset_path: Path = DEFAULT_CSV_PATH) -> list[EvaluationCase]:
    """Load and validate evaluation test cases from a `.csv` or `.json` file."""
    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ConfigurationError(f"Evaluation dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    cases: list[EvaluationCase] = []

    if suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "test_id",
                "business_profile",
                "user_query",
                "expected_intent",
                "expected_tool",
                "expected_entity",
                "expected_response_type",
            }
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ConfigurationError(
                    f"CSV dataset '{dataset_path}' must contain columns: {sorted(required)}"
                )
            for row in reader:
                cases.append(
                    EvaluationCase(
                        test_id=str(row["test_id"]).strip(),
                        business_profile=str(row["business_profile"]).strip().lower(),
                        user_query=str(row["user_query"]).strip(),
                        expected_intent=str(row["expected_intent"]).strip(),
                        expected_tool=(str(row["expected_tool"]).strip() or None),
                        expected_entity=parse_expected_entity(row["expected_entity"]),
                        expected_response_type=str(row["expected_response_type"]).strip(),
                        prior_queries=[],
                        must_include=[],
                    )
                )
        return cases

    if suffix == ".json":
        raw_list = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw_list, list):
            raise ConfigurationError("JSON evaluation dataset must be a list of case objects.")
        for item in raw_list:
            cases.append(
                EvaluationCase(
                    test_id=str(item["test_id"]).strip(),
                    business_profile=str(item["business_profile"]).strip().lower(),
                    user_query=str(item["user_query"]).strip(),
                    expected_intent=str(item["expected_intent"]).strip(),
                    expected_tool=(str(item.get("expected_tool") or "").strip() or None),
                    expected_entity=parse_expected_entity(item.get("expected_entity")),
                    expected_response_type=str(item["expected_response_type"]).strip(),
                    prior_queries=[str(q) for q in item.get("prior_queries", [])],
                    must_include=[str(m) for m in item.get("must_include", [])],
                )
            )
        return cases

    raise ConfigurationError(f"Unsupported dataset format '{suffix}'. Use .csv or .json.")


def _check_response_relevance_heuristic(
    case: EvaluationCase,
    result: WorkflowResult,
) -> bool:
    """Deterministic heuristic checking whether response addresses the query."""
    text = result.response_text.lower()
    if not text.strip():
        return False

    if case.must_include:
        return all(term.lower() in text for term in case.must_include)

    if case.expected_response_type == "not_found":
        return "not found" in text or "no order record" in text
    if case.expected_response_type == "clarification":
        return any(
            w in text
            for w in ("please share", "not sure", "clarify", "help with", "order id")
        )
    if case.expected_response_type == "refusal":
        return any(
            w in text
            for w in (
                "could not find",
                "do not have",
                "cannot",
                "privacy",
                "contact",
                "not found",
            )
        )

    # For grounded answers, check that key entities or topic words appear in response
    for val in case.expected_entity.values():
        if isinstance(val, str) and len(val) > 2:
            if val.lower() not in text and not all(tok in text for tok in val.lower().split()):
                return False

    query_keywords = [
        tok
        for tok in re.split(r"[^a-z0-9]+", case.user_query.lower())
        if len(tok) > 3
        and tok
        not in {
            "what",
            "when",
            "where",
            "which",
            "your",
            "with",
            "have",
            "this",
            "that",
            "show",
            "find",
            "under",
            "check",
            "available",
            "about",
            "please",
            "file",
        }
    ]
    if not query_keywords:
        return True
    return any(kw in text for kw in query_keywords) or (result.tool_result is not None)


@dataclass(frozen=True)
class CaseEvaluationResult:
    """Per-case deterministic evaluation breakdown."""

    test_id: str
    business_profile: str
    user_query: str
    intent_ok: bool
    entity_ok: bool
    tool_ok: bool
    tool_exec_ok: bool
    grounding_ok: bool
    relevance_ok: bool
    passed: bool
    predicted_intent: str
    expected_intent: str
    selected_tool: str | None
    expected_tool: str | None
    execution_status: str
    duration_ms: float
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_evaluation(
    workflow: UniversalAgentWorkflow,
    cases: list[EvaluationCase],
    dataset_label: str = "Day-3/test.csv",
) -> dict[str, Any]:
    """Execute all evaluation cases and compute aggregate metrics + JSON report."""
    guard = SafetyGuard()
    case_results: list[CaseEvaluationResult] = []
    tool_expected_count = 0
    tool_exec_success_count = 0

    for idx, case in enumerate(cases, start=1):
        session_id = f"eval-{case.test_id}-{idx}"
        workflow.memory.clear_session(session_id)

        # Replay prior conversation turns if evaluating multi-turn context
        for prior_q in case.prior_queries:
            workflow.run(
                prior_q,
                business_id=case.business_profile,
                session_id=session_id,
                stateless=False,
            )

        res = workflow.run(
            case.user_query,
            business_id=case.business_profile,
            session_id=session_id,
            stateless=not bool(case.prior_queries),
        )
        context = workflow.contexts[case.business_profile]

        intent_ok = res.intent.intent == case.expected_intent
        tool_ok = res.selected_tool == case.expected_tool

        actual_entities = res.entities.to_dict(omit_none=True)
        entity_ok = True
        for key, expected_val in case.expected_entity.items():
            actual_val = actual_entities.get(key)
            if isinstance(expected_val, float):
                if actual_val is None or float(actual_val) != expected_val:
                    entity_ok = False
            elif str(actual_val or "").lower() != str(expected_val).lower():
                entity_ok = False

        if case.expected_tool is not None:
            tool_expected_count += 1
            if case.expected_response_type == "not_found":
                tool_exec_ok = res.execution_status == "not_found"
            else:
                tool_exec_ok = res.execution_status == "success"
            if tool_exec_ok:
                tool_exec_success_count += 1
        else:
            tool_exec_ok = res.execution_status in {"clarification_needed", "blocked_safe", "no_tool_required"}

        grounding_ok = guard.verify_grounding_compliance(
            res.response_text,
            context,
            expected_response_type=case.expected_response_type,
        )
        relevance_ok = _check_response_relevance_heuristic(case, res)

        failures: list[str] = []
        if not intent_ok:
            failures.append(f"intent: got '{res.intent.intent}', expected '{case.expected_intent}'")
        if not entity_ok:
            failures.append(f"entity: got {actual_entities}, expected {case.expected_entity}")
        if not tool_ok:
            failures.append(f"tool: got '{res.selected_tool}', expected '{case.expected_tool}'")
        if not tool_exec_ok:
            failures.append(f"tool_exec_status: '{res.execution_status}'")
        if not grounding_ok:
            failures.append("grounding_compliance_failed")
        if not relevance_ok:
            failures.append("response_relevance_heuristic_failed")

        passed = not failures
        case_results.append(
            CaseEvaluationResult(
                test_id=case.test_id,
                business_profile=case.business_profile,
                user_query=case.user_query,
                intent_ok=intent_ok,
                entity_ok=entity_ok,
                tool_ok=tool_ok,
                tool_exec_ok=tool_exec_ok,
                grounding_ok=grounding_ok,
                relevance_ok=relevance_ok,
                passed=passed,
                predicted_intent=res.intent.intent,
                expected_intent=case.expected_intent,
                selected_tool=res.selected_tool,
                expected_tool=case.expected_tool,
                execution_status=res.execution_status,
                duration_ms=res.metrics.total_time_ms,
                failure_reasons=failures,
            )
        )

    total = len(case_results)
    passed_count = sum(1 for r in case_results if r.passed)
    failed_cases = [r.to_dict() for r in case_results if not r.passed]

    metrics_summary = {
        "intent_accuracy": round(sum(1 for r in case_results if r.intent_ok) / total, 4) if total else 0.0,
        "entity_accuracy": round(sum(1 for r in case_results if r.entity_ok) / total, 4) if total else 0.0,
        "tool_selection_accuracy": round(sum(1 for r in case_results if r.tool_ok) / total, 4) if total else 0.0,
        "tool_execution_success_rate": round(
            tool_exec_success_count / tool_expected_count, 4
        ) if tool_expected_count else 0.0,
        "grounding_compliance_rate": round(
            sum(1 for r in case_results if r.grounding_ok) / total, 4
        ) if total else 0.0,
        "response_relevance_rate": round(
            sum(1 for r in case_results if r.relevance_ok) / total, 4
        ) if total else 0.0,
        "overall_pass_rate": round(passed_count / total, 4) if total else 0.0,
    }

    return {
        "dataset": dataset_label,
        "mode": "mock_demo" if workflow.backend.is_mock else "live_gemini",
        "lcel_backend": LCEL_BACKEND,
        "evaluation_methodology": {
            "intent_entity_tool_grounding": "deterministic_exact_or_rule_check",
            "response_relevance": "deterministic_heuristic (keyword/entity/response-type match, not LLM-judged)",
        },
        "total_cases": total,
        "passed_cases": passed_count,
        "failed_cases_count": len(failed_cases),
        "metrics": metrics_summary,
        "performance_summary": workflow.metrics_collector.summarize(),
        "failures": failed_cases,
        "results": [r.to_dict() for r in case_results],
    }


def save_evaluation_report(report: dict[str, Any], output_path: Path) -> None:
    """Persist evaluation report JSON safely using UTF-8."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def format_evaluation_summary(report: dict[str, Any]) -> str:
    """Render a concise terminal summary of the evaluation report."""
    m = report["metrics"]
    perf = report["performance_summary"]
    lines = [
        f"Dataset: {report['dataset']} | Cases: {report['passed_cases']}/{report['total_cases']} passed",
        f"Mode: {report['mode']} | Relevance Method: {report['evaluation_methodology']['response_relevance']}",
        "Metrics:",
        f"  - Intent Accuracy:          {m['intent_accuracy'] * 100:.1f}%",
        f"  - Entity Accuracy:          {m['entity_accuracy'] * 100:.1f}%",
        f"  - Tool Selection Accuracy:  {m['tool_selection_accuracy'] * 100:.1f}%",
        f"  - Tool Execution Success:   {m['tool_execution_success_rate'] * 100:.1f}%",
        f"  - Grounding Compliance:     {m['grounding_compliance_rate'] * 100:.1f}%",
        f"  - Response Relevance (heur):{m['response_relevance_rate'] * 100:.1f}%",
        f"  - Overall Pass Rate:        {m['overall_pass_rate'] * 100:.1f}%",
        "Performance Summary:",
        f"  - Avg Latency: {perf['avg_total_time_ms']} ms | Total Tool Executions: {perf['total_tool_executions']} | Total LLM Calls: {perf['total_llm_calls']}",
    ]
    if report["failures"]:
        lines.append("Failures:")
        for fail in report["failures"]:
            lines.append(f"  - [{fail['test_id']}] {fail['user_query']} -> {fail['failure_reasons']}")
    else:
        lines.append("Failures: None (100% pass)")
    return "\n".join(lines)
