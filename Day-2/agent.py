"""Tasks 4, 6 & 7: Universal Tool-Calling Agent, Execution Logger, and Evaluator.

Coordinates the end-to-end Day 2 workflow:
  User Query -> Intent Classification -> Structured Entity Extraction ->
  Allowlisted Tool Selection & Validation -> Tool Execution ->
  Grounded Response Synthesis -> Bounded Memory Update -> Structured Audit Log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import time
from typing import Any

from business_config import (
    BusinessContext,
    GeminiAgentBackend,
    ValidationError,
    day1_task,
    validate_non_empty_text,
)
from business_tools import ToolRegistry, ToolResult, create_default_tool_registry
from intent_classifier import IntentResult, classify_intent
from memory_manager import ConversationMemoryManager
from structured_extraction import ExtractedEntities, extract_structured_entities


_SECRET_PATTERN = re.compile(r"(AIza[0-9A-Za-z_-]{10,}|Bearer\s+[0-9A-Za-z._-]+)")


def _redact_secrets(value: Any) -> Any:
    """Recursively mask credential-like patterns before writing execution logs."""
    if isinstance(value, str):
        return _SECRET_PATTERN.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if str(k).lower() in {"api_key", "password", "token", "secret"} else _redact_secrets(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


@dataclass(frozen=True)
class ExecutionLogEntry:
    """Structured audit record for a single agent turn."""

    timestamp_utc: str
    session_id: str
    selected_business: str
    user_query: str
    predicted_intent: str
    confidence: float
    extracted_entities: dict[str, Any]
    selected_tool: str | None
    tool_arguments: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    execution_status: str
    execution_duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolAgentLogger:
    """Collects sanitized structured execution logs for inspection and evaluation."""

    def __init__(self) -> None:
        self._entries: list[ExecutionLogEntry] = []

    def record(self, entry: ExecutionLogEntry) -> None:
        sanitized = ExecutionLogEntry(
            timestamp_utc=entry.timestamp_utc,
            session_id=entry.session_id,
            selected_business=entry.selected_business,
            user_query=str(_redact_secrets(entry.user_query)),
            predicted_intent=entry.predicted_intent,
            confidence=entry.confidence,
            extracted_entities=_redact_secrets(entry.extracted_entities),
            selected_tool=entry.selected_tool,
            tool_arguments=_redact_secrets(entry.tool_arguments),
            tool_result=_redact_secrets(entry.tool_result),
            execution_status=entry.execution_status,
            execution_duration_ms=entry.execution_duration_ms,
        )
        self._entries.append(sanitized)

    def get_logs(self) -> list[ExecutionLogEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()


@dataclass(frozen=True)
class AgentResponse:
    """Complete structured output returned by UniversalToolAgent.handle_query()."""

    response_text: str
    business_id: str
    business_name: str
    session_id: str
    intent: IntentResult
    entities: ExtractedEntities
    selected_tool: str | None
    tool_arguments: dict[str, Any] | None
    tool_result: ToolResult | None
    execution_status: str
    duration_ms: float
    is_mock: bool


class UniversalToolAgent:
    """Reusable, config-driven AI agent with allowlisted tools and bounded memory."""

    def __init__(
        self,
        context: BusinessContext,
        backend: GeminiAgentBackend,
        registry: ToolRegistry | None = None,
        memory: ConversationMemoryManager | None = None,
        logger: ToolAgentLogger | None = None,
    ) -> None:
        self.context = context
        self.backend = backend
        self.registry = registry or create_default_tool_registry()
        self.memory = memory or ConversationMemoryManager(window_size=6)
        self.logger = logger or ToolAgentLogger()

    def switch_business(self, new_context: BusinessContext) -> None:
        """Switch active business configuration at runtime without altering core logic."""
        self.context = new_context

    def select_tool(
        self,
        intent_result: IntentResult,
        entities: ExtractedEntities,
        user_query: str,
    ) -> tuple[str | None, dict[str, Any]]:
        """Map validated intent + extracted entities to an allowlisted tool and arguments."""
        intent = intent_result.intent
        lower_q = user_query.lower()

        if intent in {"product_inquiry", "pricing_inquiry", "course_information"}:
            search_query = entities.product_name or entities.course_name or ""
            if not search_query:
                # Fall back to meaningful tokens when no exact phrase was extracted
                if "hoodie" in lower_q:
                    search_query = "hoodie"
                elif "course" in lower_q or "bootcamp" in lower_q:
                    search_query = ""
            sort_by = "price_asc" if any(w in lower_q for w in ("cheapest", "lowest price", "least expensive")) else None
            args = {
                "query": search_query,
                "category": entities.product_category,
                "max_price": entities.price,
                "sort_by": sort_by,
            }
            return self._ensure_allowed("product_search"), args

        if intent == "order_status":
            if not entities.order_id:
                return None, {}
            return self._ensure_allowed("order_lookup"), {"order_id": entities.order_id}

        if intent == "business_hours":
            return self._ensure_allowed("business_info"), {"topic": "hours"}

        if intent == "contact_information":
            return self._ensure_allowed("business_info"), {"topic": "contact"}

        if intent in {"refund_request", "cancellation_request"}:
            return self._ensure_allowed("business_info"), {"topic": "refund"}

        if intent == "appointment_booking":
            return self._ensure_allowed("appointment_availability"), {
                "date": entities.date or "",
                "topic": user_query,
            }

        if intent in {"general_inquiry", "complaint"}:
            return self._ensure_allowed("knowledge_base_search"), {"query": user_query}

        return None, {}

    def _ensure_allowed(self, tool_name: str) -> str:
        if not self.registry.is_allowed(tool_name):
            raise ValidationError(f"Selected tool '{tool_name}' is not registered in ToolRegistry.")
        return tool_name

    def _format_grounded_demo_reply(
        self,
        user_query: str,
        intent_result: IntentResult,
        tool_result: ToolResult | None,
    ) -> str:
        """Build a clear, grounded response directly from deterministic tool output."""
        prefix = f"[MOCK DEMO | {self.context.business_name}]"
        lower_q = user_query.lower()

        if intent_result.intent == "order_status" and tool_result is None:
            return (
                f"{prefix} Please share your order ID (for example, `DZ1024` or `ES2041`) "
                "so I can look up its status for you."
            )

        if intent_result.intent == "unknown" or tool_result is None:
            topics = ", ".join(self.context.profile.support_topics)
            return (
                f"{prefix} I am not sure I understood your request clearly. "
                f"I can help with {topics} for {self.context.business_name}, or you can contact "
                f"{self.context.contact}."
            )

        data = tool_result.data
        tool = tool_result.tool_name

        if tool_result.status == "validation_error":
            return f"{prefix} I could not process that request: {tool_result.message}"

        if tool == "product_search":
            items = data.get("items", [])
            if not items:
                return (
                    f"{prefix} I could not find matching items in the {self.context.business_name} "
                    "catalog for your filters."
                )
            cheapest_mode = any(w in lower_q for w in ("cheapest", "lowest price", "least expensive"))
            if cheapest_mode:
                first = items[0]
                details = ", ".join(
                    f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                    for k, v in first.items()
                )
                return f"{prefix} The cheapest matching option is:\n- {details}"

            lines = [f"{prefix} Found {len(items)} matching item(s) in {self.context.business_name}:"]
            for item in items:
                details = "; ".join(
                    f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                    for k, v in item.items()
                )
                lines.append(f"- {details}")
            return "\n".join(lines)

        if tool == "order_lookup":
            if not data.get("found"):
                return (
                    f"{prefix} {data.get('message', 'Order not found.')} "
                    "Please verify your order ID or contact "
                    f"{self.context.contact}."
                )
            order = data["order"]
            return (
                f"{prefix} (Mock Order Lookup) Order `{order['order_id']}` for "
                f"**{order['item']}** is currently **{order['status']}**. "
                f"Estimated delivery/access: {order['estimated_delivery']} "
                f"(Carrier/Portal: {order['carrier']})."
            )

        if tool == "business_info":
            lines = [f"{prefix} Verified {self.context.business_name} information:"]
            lines.append(f"- Support Hours: {data.get('support_hours')}")
            lines.append(f"- Contact: {data.get('contact')}")
            policies = data.get("policies", {})
            for p_name, p_text in policies.items():
                lines.append(f"- {p_name.replace('_', ' ').title()}: {p_text}")
            if intent_result.intent in {"refund_request", "cancellation_request"}:
                lines.append(
                    "- Note: I cannot automatically process refunds or cancellations here; "
                    f"please contact {self.context.contact} with your order ID."
                )
            return "\n".join(lines)

        if tool == "appointment_availability":
            slots = data.get("slots", [])
            if not slots:
                return (
                    f"{prefix} No available appointment slots matched your filter. "
                    f"Please reach out to {self.context.contact}."
                )
            lines = [
                f"{prefix} Available slots (preview only; no booking has been made):"
            ]
            for slot in slots:
                lines.append(
                    f"- [{slot['slot_id']}] {slot['date']} at {slot['time']} — {slot['topic']}"
                )
            return "\n".join(lines)

        if tool == "knowledge_base_search":
            faqs = data.get("faqs", [])
            policies = data.get("policies", {})
            lines = [f"{prefix} Here is what I found in the {self.context.business_name} knowledge base:"]
            if intent_result.intent == "complaint":
                lines.append(
                    f"- We apologize for the inconvenience. Please email {self.context.contact} "
                    f"during {self.context.support_hours} for priority resolution."
                )
            for faq in faqs[:2]:
                lines.append(f"- FAQ: {faq['question']} -> {faq['answer']}")
            for p_key, p_val in list(policies.items())[:2]:
                lines.append(f"- {p_key.replace('_', ' ').title()}: {p_val}")
            if len(lines) == 1:
                lines.append(
                    f"- I do not have enough information on that specific topic. Contact {self.context.contact}."
                )
            return "\n".join(lines)

        return f"{prefix} Completed tool `{tool}`."

    def _synthesize_live_reply(
        self,
        user_query: str,
        intent_result: IntentResult,
        tool_result: ToolResult | None,
        session_id: str,
        use_memory: bool,
    ) -> str:
        """Use Gemini in live mode to turn verified tool results into a natural answer."""
        base_system = day1_task.build_grounded_system_instruction(self.context.profile)
        tool_json = json.dumps(tool_result.to_dict(), indent=2) if tool_result else "None"
        system_instruction = (
            f"{base_system}\n\n"
            "DAY 2 TOOL EXECUTION RULES:\n"
            "1. Base your answer strictly on the Verified Tool Result JSON below.\n"
            "2. NEVER invent order records, prices, or policies not present in the tool result.\n"
            "3. If a mock order lookup or appointment availability preview ran, NEVER claim a real "
            "appointment was booked or a refund was executed.\n"
            f"Predicted Intent: {intent_result.intent} (confidence={intent_result.confidence:.2f})\n"
            f"Verified Tool Result JSON:\n{tool_json}"
        )
        history = self.memory.get_day1_turns(session_id) if use_memory else None
        llm_res = self.backend.generate(
            user_prompt=user_query,
            system_instruction=system_instruction,
            temperature=0.2,
            history=history,
            profile=self.context.profile,
        )
        return llm_res.text

    def handle_query(
        self,
        user_query: str,
        session_id: str = "default",
        stateless: bool = False,
    ) -> AgentResponse:
        """Run the full Day 2 agent pipeline for a single user query."""
        cleaned_query = validate_non_empty_text(user_query, "user_query")
        cleaned_sid = validate_non_empty_text(session_id, "session_id")
        start = time.perf_counter()

        use_memory = (not stateless) and (not self.memory.is_stateless)
        recent_texts = self.memory.get_recent_texts(cleaned_sid) if use_memory else []

        intent_result = classify_intent(
            cleaned_query,
            backend=self.backend,
            context=self.context,
            recent_history=recent_texts,
        )
        entities = extract_structured_entities(
            cleaned_query,
            backend=self.backend,
            context=self.context,
            recent_history=recent_texts,
        )

        selected_tool, tool_args = self.select_tool(intent_result, entities, cleaned_query)

        tool_result: ToolResult | None = None
        if selected_tool is not None:
            tool_result = self.registry.execute_tool(selected_tool, tool_args, self.context)
            exec_status = tool_result.status
        elif intent_result.requires_clarification or intent_result.intent == "unknown":
            exec_status = "clarification_needed"
        else:
            exec_status = "no_tool_required"

        if self.backend.is_mock:
            response_text = self._format_grounded_demo_reply(
                cleaned_query,
                intent_result,
                tool_result,
            )
        else:
            response_text = self._synthesize_live_reply(
                cleaned_query,
                intent_result,
                tool_result,
                session_id=cleaned_sid,
                use_memory=use_memory,
            )

        if use_memory:
            self.memory.add_message(
                cleaned_sid,
                role="user",
                text=cleaned_query,
                intent=intent_result.intent,
                entities=entities.to_dict(omit_none=True),
            )
            self.memory.add_message(
                cleaned_sid,
                role="assistant",
                text=response_text,
                intent=intent_result.intent,
            )

        total_ms = round((time.perf_counter() - start) * 1000, 2)

        self.logger.record(
            ExecutionLogEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                session_id=cleaned_sid if use_memory else f"{cleaned_sid}(stateless)",
                selected_business=self.context.business_id,
                user_query=cleaned_query,
                predicted_intent=intent_result.intent,
                confidence=round(intent_result.confidence, 2),
                extracted_entities=entities.to_dict(omit_none=True),
                selected_tool=selected_tool,
                tool_arguments=tool_args if selected_tool else None,
                tool_result=tool_result.to_dict() if tool_result else None,
                execution_status=exec_status,
                execution_duration_ms=total_ms,
            )
        )

        return AgentResponse(
            response_text=response_text,
            business_id=self.context.business_id,
            business_name=self.context.business_name,
            session_id=cleaned_sid,
            intent=intent_result,
            entities=entities,
            selected_tool=selected_tool,
            tool_arguments=tool_args if selected_tool else None,
            tool_result=tool_result,
            execution_status=exec_status,
            duration_ms=total_ms,
            is_mock=self.backend.is_mock,
        )


def evaluate_agent_cases(
    cases: list[dict[str, Any]],
    contexts: dict[str, BusinessContext],
    backend: GeminiAgentBackend,
) -> dict[str, Any]:
    """Task 7: Evaluate intent accuracy, entity extraction, and tool selection."""
    total = len(cases)
    intent_correct = 0
    tool_correct = 0
    entity_correct = 0
    details: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        biz_id = str(case["business_id"]).lower()
        query = str(case["query"])
        expected_intent = str(case["expected_intent"])
        expected_tool = case.get("expected_tool")
        expected_entities = case.get("expected_entities", {})

        agent = UniversalToolAgent(
            context=contexts[biz_id],
            backend=backend,
            memory=ConversationMemoryManager(window_size=0),
        )
        res = agent.handle_query(query, session_id=f"eval-{idx}", stateless=True)

        i_ok = res.intent.intent == expected_intent
        t_ok = res.selected_tool == expected_tool
        actual_entities = res.entities.to_dict(omit_none=True)
        e_ok = all(actual_entities.get(k) == v for k, v in expected_entities.items())

        intent_correct += int(i_ok)
        tool_correct += int(t_ok)
        entity_correct += int(e_ok)

        details.append(
            {
                "query": query,
                "business_id": biz_id,
                "intent_ok": i_ok,
                "predicted_intent": res.intent.intent,
                "expected_intent": expected_intent,
                "tool_ok": t_ok,
                "selected_tool": res.selected_tool,
                "expected_tool": expected_tool,
                "entity_ok": e_ok,
            }
        )

    return {
        "total_cases": total,
        "intent_accuracy": round(intent_correct / total, 2) if total else 0.0,
        "tool_accuracy": round(tool_correct / total, 2) if total else 0.0,
        "entity_accuracy": round(entity_correct / total, 2) if total else 0.0,
        "details": details,
    }
