"""Task 7: Performance, Latency, and Token Usage Metrics for Day 3.

Uses `time.perf_counter()` to measure per-stage execution duration and aggregates
actual Gemini `usage_metadata` when available, clearly distinguishing estimates
from live API-reported counts without fabricating metrics or logging secrets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any


def estimate_token_count(text: str) -> int:
    """Rough local token estimate (~4 characters per token) for offline budgeting."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class StageTimer:
    """Context manager measuring elapsed wall-clock milliseconds via perf_counter()."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 3)


@dataclass(frozen=True)
class TokenUsageMetrics:
    """Separates local estimates from authoritative Gemini usage_metadata."""

    estimated_prompt_tokens: int
    estimated_response_tokens: int
    api_prompt_tokens: int | None = None
    api_candidates_tokens: int | None = None
    api_total_tokens: int | None = None
    usage_source: str = "estimated_only_no_live_api_call"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowMetrics:
    """Per-request timing, tool count, LLM call count, and token metrics."""

    business_id: str
    intent: str
    selected_tool: str | None
    status: str
    total_time_ms: float
    intent_classification_ms: float
    entity_extraction_ms: float
    knowledge_retrieval_ms: float
    tool_execution_ms: float
    response_generation_ms: float
    tool_execution_count: int
    llm_call_count: int
    token_usage: TokenUsageMetrics

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_usage"] = self.token_usage.to_dict()
        return payload


@dataclass
class MetricsCollector:
    """Aggregates WorkflowMetrics records across interactive or batch runs."""

    records: list[WorkflowMetrics] = field(default_factory=list)

    def record(self, metrics: WorkflowMetrics) -> None:
        self.records.append(metrics)

    def clear(self) -> None:
        self.records.clear()

    def summarize(self) -> dict[str, Any]:
        """Compute aggregate latency, tool usage, LLM call, and token summaries."""
        count = len(self.records)
        if count == 0:
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "avg_total_time_ms": 0.0,
                "avg_intent_ms": 0.0,
                "avg_extraction_ms": 0.0,
                "avg_retrieval_ms": 0.0,
                "avg_tool_ms": 0.0,
                "avg_generation_ms": 0.0,
                "total_tool_executions": 0,
                "total_llm_calls": 0,
                "total_estimated_prompt_tokens": 0,
                "total_estimated_response_tokens": 0,
                "api_reported_total_tokens": None,
            }

        success_statuses = {"success", "clarification_needed", "blocked_safe", "not_found"}
        successful = sum(1 for r in self.records if r.status in success_statuses)
        api_totals = [
            r.token_usage.api_total_tokens
            for r in self.records
            if r.token_usage.api_total_tokens is not None
        ]

        return {
            "total_runs": count,
            "successful_runs": successful,
            "avg_total_time_ms": round(sum(r.total_time_ms for r in self.records) / count, 3),
            "avg_intent_ms": round(
                sum(r.intent_classification_ms for r in self.records) / count, 3
            ),
            "avg_extraction_ms": round(
                sum(r.entity_extraction_ms for r in self.records) / count, 3
            ),
            "avg_retrieval_ms": round(
                sum(r.knowledge_retrieval_ms for r in self.records) / count, 3
            ),
            "avg_tool_ms": round(sum(r.tool_execution_ms for r in self.records) / count, 3),
            "avg_generation_ms": round(
                sum(r.response_generation_ms for r in self.records) / count, 3
            ),
            "total_tool_executions": sum(r.tool_execution_count for r in self.records),
            "total_llm_calls": sum(r.llm_call_count for r in self.records),
            "total_estimated_prompt_tokens": sum(
                r.token_usage.estimated_prompt_tokens for r in self.records
            ),
            "total_estimated_response_tokens": sum(
                r.token_usage.estimated_response_tokens for r in self.records
            ),
            "api_reported_total_tokens": sum(api_totals) if api_totals else None,
        }
