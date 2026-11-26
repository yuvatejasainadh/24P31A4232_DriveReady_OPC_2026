"""Tasks 2 & 3: Complete Multi-Step Agent Workflow & Execution Paradigm Comparison.

Connects Day 1 (Gemini SDK + profiles), Day 2 (UniversalToolAgent, ToolRegistry,
ConversationMemoryManager, ToolAgentLogger), and Day 3 (LCEL chains,
KnowledgeRetriever, SafetyGuard, MetricsCollector) without duplicating core code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

DAY2_DIR = Path(__file__).resolve().parent.parent / "Day-2"
if str(DAY2_DIR) not in sys.path:
    sys.path.append(str(DAY2_DIR))

from agent import ExecutionLogEntry, ToolAgentLogger, UniversalToolAgent  # noqa: E402
from business_config import (  # noqa: E402
    BusinessContext,
    GeminiAgentBackend,
    ValidationError,
    day1_task,
    validate_non_empty_text,
)
from business_tools import ToolRegistry, ToolResult, create_default_tool_registry  # noqa: E402
from chains import (  # noqa: E402
    LCEL_BACKEND,
    build_final_formatting_chain,
    build_master_lcel_pipeline,
    build_response_generation_chain,
    summarize_policy_via_lcel,
)
from grounding import KnowledgeRetriever, RetrievalResult, SafetyGuard  # noqa: E402
from intent_classifier import IntentResult  # noqa: E402
from memory_manager import ConversationMemoryManager  # noqa: E402
from metrics import (  # noqa: E402
    MetricsCollector,
    StageTimer,
    TokenUsageMetrics,
    WorkflowMetrics,
    estimate_token_count,
)
from structured_extraction import ExtractedEntities  # noqa: E402


@dataclass(frozen=True)
class WorkflowResult:
    """Complete structured output from running UniversalAgentWorkflow."""

    response_text: str
    business_id: str
    business_name: str
    session_id: str
    intent: IntentResult
    entities: ExtractedEntities
    retrieval: RetrievalResult | None
    selected_tool: str | None
    tool_arguments: dict[str, Any] | None
    tool_result: ToolResult | None
    execution_status: str
    safety_code: str
    metrics: WorkflowMetrics
    lcel_backend: str
    is_mock: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "business_id": self.business_id,
            "business_name": self.business_name,
            "session_id": self.session_id,
            "intent": self.intent.to_dict(),
            "entities": self.entities.to_dict(omit_none=True),
            "selected_tool": self.selected_tool,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "execution_status": self.execution_status,
            "safety_code": self.safety_code,
            "metrics": self.metrics.to_dict(),
            "lcel_backend": self.lcel_backend,
            "is_mock": self.is_mock,
        }


class UniversalAgentWorkflow:
    """Multi-step LCEL-orchestrated Universal Business AI Agent."""

    def __init__(
        self,
        contexts: dict[str, BusinessContext],
        default_business_id: str,
        backend: GeminiAgentBackend,
        registry: ToolRegistry | None = None,
        memory: ConversationMemoryManager | None = None,
        logger: ToolAgentLogger | None = None,
        retriever: KnowledgeRetriever | None = None,
        safety_guard: SafetyGuard | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        if default_business_id not in contexts:
            raise ValidationError(f"Unknown default business '{default_business_id}'.")

        self.contexts = contexts
        self.active_business_id = default_business_id
        self.backend = backend
        self.registry = registry or create_default_tool_registry()
        self.memory = memory or ConversationMemoryManager(window_size=6)
        self.logger = logger or ToolAgentLogger()
        self.retriever = retriever or KnowledgeRetriever()
        self.safety_guard = safety_guard or SafetyGuard()
        self.metrics_collector = metrics_collector or MetricsCollector()

        # Reuse Day 2's UniversalToolAgent for tool selection & grounded demo formatting
        self.day2_agent = UniversalToolAgent(
            context=contexts[default_business_id],
            backend=backend,
            registry=self.registry,
            memory=self.memory,
            logger=self.logger,
        )

        self._pre_pipeline = build_master_lcel_pipeline(
            contexts=self.contexts,
            backend=self.backend,
            memory_manager=self.memory,
            retriever=self.retriever,
        )
        self._response_chain = build_response_generation_chain(self.backend)
        self._format_chain = build_final_formatting_chain()

    @property
    def active_context(self) -> BusinessContext:
        return self.contexts[self.active_business_id]

    def switch_business(self, business_id: str) -> BusinessContext:
        cleaned = validate_non_empty_text(business_id, "business_id").lower()
        if cleaned not in self.contexts:
            raise ValidationError(
                f"Unknown business profile '{cleaned}'. Available: {', '.join(sorted(self.contexts))}"
            )
        self.active_business_id = cleaned
        self.day2_agent.switch_business(self.contexts[cleaned])
        return self.contexts[cleaned]

    def _refine_tool_args(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any],
        user_query: str,
        context: BusinessContext,
    ) -> dict[str, Any]:
        """Preserve specific item queries (e.g., 'sneakers') so missing items return 0 matches."""
        if tool_name != "product_search":
            return tool_args

        current_q = str(tool_args.get("query", "")).strip()
        if current_q:
            return tool_args

        lower_q = user_query.lower()
        # Keep broad catalog overview queries open, but pass specific nouns like "sneakers" or "python"
        if any(phrase in lower_q for phrase in ("what products", "what courses", "show all", "cheapest")):
            return tool_args

        tokens = [
            tok
            for tok in re.split(r"[^a-z0-9]+", lower_q)
            if len(tok) > 2
            and tok
            not in {
                "the",
                "for",
                "and",
                "are",
                "you",
                "your",
                "can",
                "show",
                "find",
                "sell",
                "available",
                "what",
                "have",
                "offer",
                "does",
                "with",
                "about",
                "course",
                "courses",
                "product",
                "products",
                "fee",
                "price",
                "under",
                context.business_id,
            }
        ]
        if tokens:
            return {**tool_args, "query": " ".join(tokens)}
        return tool_args

    def run(
        self,
        user_query: str,
        business_id: str | None = None,
        session_id: str = "default",
        stateless: bool = False,
    ) -> WorkflowResult:
        """Execute the full multi-step Day 3 workflow for a single user query."""
        cleaned_query = validate_non_empty_text(user_query, "user_query")
        cleaned_sid = validate_non_empty_text(session_id, "session_id")
        target_biz = (business_id or self.active_business_id).strip().lower()
        if target_biz not in self.contexts:
            raise ValidationError(
                f"Unknown business profile '{target_biz}'. Available: {', '.join(sorted(self.contexts))}"
            )
        if target_biz != self.active_business_id:
            self.switch_business(target_biz)
        context = self.contexts[target_biz]

        with StageTimer() as total_timer:
            # 1. Safety Gate Pre-Check (injection, cross-customer data, unsupported claims)
            safety = self.safety_guard.check_query(cleaned_query, context)
            if not safety.allowed:
                empty_intent = IntentResult(
                    intent="unknown",
                    confidence=1.0,
                    entities={},
                    requires_clarification=False,
                    reason=f"Blocked by safety guard: {safety.reason_code}",
                    is_mock=self.backend.is_mock,
                )
                empty_entities = ExtractedEntities()
                metrics = WorkflowMetrics(
                    business_id=context.business_id,
                    intent=empty_intent.intent,
                    selected_tool=None,
                    status="blocked_safe",
                    total_time_ms=total_timer.elapsed_ms,
                    intent_classification_ms=0.0,
                    entity_extraction_ms=0.0,
                    knowledge_retrieval_ms=0.0,
                    tool_execution_ms=0.0,
                    response_generation_ms=0.0,
                    tool_execution_count=0,
                    llm_call_count=0,
                    token_usage=TokenUsageMetrics(
                        estimated_prompt_tokens=estimate_token_count(cleaned_query),
                        estimated_response_tokens=estimate_token_count(safety.safe_message),
                    ),
                )
                self.metrics_collector.record(metrics)
                return WorkflowResult(
                    response_text=safety.safe_message,
                    business_id=context.business_id,
                    business_name=context.business_name,
                    session_id=cleaned_sid,
                    intent=empty_intent,
                    entities=empty_entities,
                    retrieval=None,
                    selected_tool=None,
                    tool_arguments=None,
                    tool_result=None,
                    execution_status="blocked_safe",
                    safety_code=safety.reason_code,
                    metrics=metrics,
                    lcel_backend=LCEL_BACKEND,
                    is_mock=self.backend.is_mock,
                )

            # 2. Context Prep + Intent/Entities + Retrieval via LCEL Pipeline
            with StageTimer() as pre_timer:
                state = self._pre_pipeline.invoke(
                    {
                        "query": cleaned_query,
                        "business_id": target_biz,
                        "session_id": cleaned_sid,
                        "stateless": stateless,
                    }
                )

            intent_res: IntentResult = state["intent_result"]
            entities: ExtractedEntities = state["entities"]
            retrieval: RetrievalResult = state["retrieval"]
            knowledge_context: str = state["knowledge_context"]
            use_memory = not bool(state["stateless"])

            # Split stage estimate cleanly across intent/extraction/retrieval
            intent_ms = round(pre_timer.elapsed_ms * 0.45, 3)
            extract_ms = round(pre_timer.elapsed_ms * 0.35, 3)
            retrieval_ms = round(pre_timer.elapsed_ms * 0.20, 3)

            # 3. Decision Engine -> Tool Selection -> Validation -> Execution
            selected_tool, tool_args = self.day2_agent.select_tool(
                intent_res, entities, cleaned_query
            )
            tool_args = self._refine_tool_args(selected_tool, tool_args, cleaned_query, context)

            tool_result: ToolResult | None = None
            tool_exec_count = 0
            with StageTimer() as tool_timer:
                if selected_tool is not None:
                    tool_result = self.registry.execute_tool(selected_tool, tool_args, context)
                    tool_exec_count = 1
                    exec_status = tool_result.status
                elif intent_res.requires_clarification or intent_res.intent == "unknown":
                    exec_status = "clarification_needed"
                else:
                    exec_status = "no_tool_required"

            # 4. Grounded Response Generation & Formatting (LCEL Chains 4 & 5)
            llm_calls = int(state.get("analysis_llm_calls", 0))
            with StageTimer() as gen_timer:
                if self.backend.is_mock:
                    if selected_tool == "knowledge_base_search" and intent_res.intent != "complaint":
                        if retrieval.has_grounding_facts:
                            raw_answer = f"[MOCK DEMO | {context.business_name}] {retrieval.to_context_text()}"
                        else:
                            raw_answer = (
                                f"[MOCK DEMO | {context.business_name}] I do not have enough information "
                                f"in the verified {context.business_name} knowledge base to answer that "
                                f"reliably. Please contact {context.contact}."
                            )
                    else:
                        raw_answer = self.day2_agent._format_grounded_demo_reply(
                            cleaned_query, intent_res, tool_result
                        )
                else:
                    tool_text = (
                        json.dumps(tool_result.to_dict(), indent=2)
                        if tool_result is not None
                        else "None"
                    )
                    raw_answer = self._response_chain.invoke(
                        {
                            "business_name": context.business_name,
                            "industry": context.industry,
                            "tone": context.profile.tone,
                            "knowledge_context": knowledge_context,
                            "tool_result_text": tool_text,
                            "query": cleaned_query,
                        }
                    )
                    llm_calls += 1

                formatted = self._format_chain.invoke(
                    {
                        "raw_response": raw_answer,
                        "is_mock": self.backend.is_mock,
                        "context": context,
                    }
                )
                final_text: str = formatted["final_response"]

            # 5. Bounded Memory Update
            if use_memory:
                self.memory.add_message(
                    cleaned_sid,
                    role="user",
                    text=cleaned_query,
                    intent=intent_res.intent,
                    entities=entities.to_dict(omit_none=True),
                )
                self.memory.add_message(
                    cleaned_sid,
                    role="assistant",
                    text=final_text,
                    intent=intent_res.intent,
                )

        # 6. Metrics & Audit Log Recording
        prompt_text_for_est = "\n".join(
            [cleaned_query, knowledge_context, json.dumps(tool_result.data) if tool_result else ""]
        )
        token_usage = TokenUsageMetrics(
            estimated_prompt_tokens=estimate_token_count(prompt_text_for_est),
            estimated_response_tokens=estimate_token_count(final_text),
            api_prompt_tokens=None,
            api_candidates_tokens=None,
            api_total_tokens=None,
            usage_source=(
                "estimated_only_no_live_api_call"
                if self.backend.is_mock
                else "gemini_lcel_estimated_or_api"
            ),
        )
        wf_metrics = WorkflowMetrics(
            business_id=context.business_id,
            intent=intent_res.intent,
            selected_tool=selected_tool,
            status=exec_status,
            total_time_ms=total_timer.elapsed_ms,
            intent_classification_ms=intent_ms,
            entity_extraction_ms=extract_ms,
            knowledge_retrieval_ms=retrieval_ms,
            tool_execution_ms=tool_timer.elapsed_ms,
            response_generation_ms=gen_timer.elapsed_ms,
            tool_execution_count=tool_exec_count,
            llm_call_count=llm_calls,
            token_usage=token_usage,
        )
        self.metrics_collector.record(wf_metrics)

        self.logger.record(
            ExecutionLogEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                session_id=cleaned_sid if use_memory else f"{cleaned_sid}(stateless)",
                selected_business=context.business_id,
                user_query=cleaned_query,
                predicted_intent=intent_res.intent,
                confidence=round(intent_res.confidence, 2),
                extracted_entities=entities.to_dict(omit_none=True),
                selected_tool=selected_tool,
                tool_arguments=tool_args if selected_tool else None,
                tool_result=tool_result.to_dict() if tool_result else None,
                execution_status=exec_status,
                execution_duration_ms=wf_metrics.total_time_ms,
            )
        )

        return WorkflowResult(
            response_text=final_text,
            business_id=context.business_id,
            business_name=context.business_name,
            session_id=cleaned_sid,
            intent=intent_res,
            entities=entities,
            retrieval=retrieval,
            selected_tool=selected_tool,
            tool_arguments=tool_args if selected_tool else None,
            tool_result=tool_result,
            execution_status=exec_status,
            safety_code="ok",
            metrics=wf_metrics,
            lcel_backend=LCEL_BACKEND,
            is_mock=self.backend.is_mock,
        )


# ---------------------------------------------------------------------------
# Task 3: Deterministic Logic vs. LLM Processing vs. Agent Execution
# ---------------------------------------------------------------------------


def calculate_discount(price_inr: float, discount_percent: float) -> float:
    """Deterministic Python business math (exact, zero LLM cost)."""
    if price_inr < 0 or not 0 <= discount_percent <= 100:
        raise ValidationError("Invalid price or discount percentage.")
    return round(price_inr * (1.0 - discount_percent / 100.0), 2)


def is_within_budget(price_inr: float, max_budget_inr: float) -> bool:
    """Deterministic numeric comparison."""
    return price_inr <= max_budget_inr


def validate_order_id_format(order_id: str) -> bool:
    """Deterministic format validation for order IDs like DZ1024."""
    return bool(re.match(r"^[A-Z]{2,4}\d{3,6}$", order_id.strip().upper()))


def compare_execution_approaches(
    workflow: UniversalAgentWorkflow,
    context: BusinessContext,
) -> dict[str, Any]:
    """Task 3: Compare Deterministic Code vs. LLM-Only vs. Tool-Using Agent."""
    # A. Deterministic Python Logic
    with StageTimer() as det_timer:
        discounted_price = calculate_discount(1799.0, 10.0)
        budget_ok = is_within_budget(899.0, 1000.0)
        valid_id = validate_order_id_format("DZ1024")
    deterministic_summary = {
        "approach": "Deterministic Python Logic",
        "operations": {
            "apply_10_percent_discount_on_1799": discounted_price,
            "is_899_within_1000_budget": budget_ok,
            "is_DZ1024_valid_format": valid_id,
        },
        "duration_ms": det_timer.elapsed_ms,
        "best_for": "Exact math, threshold checks, schema/regex validation, zero latency/cost.",
    }

    # B. LLM-Only Processing (LCEL prompt -> model -> StrOutputParser, no tool access)
    with StageTimer() as llm_timer:
        policy_summary = summarize_policy_via_lcel(
            workflow.backend,
            context,
            policy_topic="returns_and_exchanges"
            if context.business_id == "dripzoid"
            else "refund_policy",
        )
        llm_order_limitation = (
            "[MOCK DEMO | LLM-Only] Cannot verify live order DZ1024 because no order_lookup tool is connected."
            if workflow.backend.is_mock
            else day1_task.GeminiAgentBackend(force_demo=False)
            .generate(
                user_prompt="Where is my order DZ1024?",
                system_instruction="Never invent order records. If you lack order lookup tools, say so.",
                temperature=0.0,
            )
            .text
        )
    llm_summary = {
        "approach": "LLM-Only Processing (LCEL Prompt -> Model -> Parser)",
        "policy_summary": policy_summary,
        "order_query_without_tools": llm_order_limitation,
        "duration_ms": llm_timer.elapsed_ms,
        "best_for": "Natural-language comprehension, summarization, tone adaptation.",
    }

    # C. Agent-Based Processing (Intent + Extraction + Tool Execution + Grounded Reply)
    agent_res = workflow.run(
        "Where is my order DZ1024?" if context.business_id == "dripzoid" else "Where is my order ES2041?",
        business_id=context.business_id,
        session_id="compare-exec",
        stateless=True,
    )
    agent_summary = {
        "approach": "Tool-Using AI Agent (Multi-Step Workflow)",
        "query": "Where is my order DZ1024?" if context.business_id == "dripzoid" else "Where is my order ES2041?",
        "predicted_intent": agent_res.intent.intent,
        "extracted_entities": agent_res.entities.to_dict(omit_none=True),
        "selected_tool": agent_res.selected_tool,
        "tool_status": agent_res.execution_status,
        "grounded_response": agent_res.response_text,
        "duration_ms": agent_res.metrics.total_time_ms,
        "best_for": "Multi-step reasoning requiring real/mock system lookups, routing, and grounding.",
    }

    return {
        "deterministic": deterministic_summary,
        "llm_only": llm_summary,
        "agent": agent_summary,
    }
