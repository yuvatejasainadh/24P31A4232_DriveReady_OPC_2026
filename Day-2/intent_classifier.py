"""Task 1: Business Intent Classification for Day 2.

Classifies user queries into configurable business intents with confidence
scores and clarification flags. Uses Gemini in live mode and deterministic
rule matching in demo mode. Never executes business actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from business_config import (
    BusinessContext,
    GeminiAgentBackend,
    ValidationError,
    validate_non_empty_text,
)
from structured_extraction import extract_structured_entities


DEFAULT_INTENTS: tuple[str, ...] = (
    "product_inquiry",
    "order_status",
    "pricing_inquiry",
    "refund_request",
    "cancellation_request",
    "appointment_booking",
    "course_information",
    "business_hours",
    "contact_information",
    "complaint",
    "general_inquiry",
    "unknown",
)


@dataclass(frozen=True)
class IntentResult:
    """Structured output returned by business intent classification."""

    intent: str
    confidence: float
    entities: dict[str, Any]
    requires_clarification: bool
    reason: str = ""
    is_mock: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize core intent fields matching the Day 2 JSON contract."""
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 2),
            "entities": dict(self.entities),
            "requires_clarification": self.requires_clarification,
        }

    def to_full_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _resolve_allowed_intents(
    context: BusinessContext | None,
    allowed_intents: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if allowed_intents is not None:
        cleaned = [str(i).strip() for i in allowed_intents if str(i).strip()]
        if "unknown" not in cleaned:
            cleaned.append("unknown")
        return cleaned
    if context is not None and context.supported_intents:
        intents = list(context.supported_intents)
        if "unknown" not in intents:
            intents.append("unknown")
        return intents
    return list(DEFAULT_INTENTS)


def _classify_deterministic(
    message: str,
    allowed_intents: list[str],
    context: BusinessContext | None,
    recent_history: list[str] | None,
) -> IntentResult:
    """Deterministic intent classifier for offline demo mode."""
    lower = message.lower().strip()
    entities_obj = extract_structured_entities(
        message,
        backend=None,
        context=context,
        recent_history=recent_history,
    )
    compact_entities = entities_obj.to_dict(omit_none=True)

    # Guard against meaningless / non-actionable / too-short gibberish queries
    alpha_tokens = re.findall(r"[a-z]+", lower)
    if not alpha_tokens or (len(lower) <= 3 and lower not in {"hi", "hey"}):
        return IntentResult(
            intent="unknown",
            confidence=0.25,
            entities=compact_entities,
            requires_clarification=True,
            reason="Query lacks recognizable intent keywords.",
            is_mock=True,
        )

    def pick(candidate: str, conf: float, reason: str, clarify: bool = False) -> IntentResult:
        chosen = candidate if candidate in allowed_intents else (
            "product_inquiry" if candidate == "course_information" and "product_inquiry" in allowed_intents
            else ("general_inquiry" if "general_inquiry" in allowed_intents else "unknown")
        )
        return IntentResult(
            intent=chosen,
            confidence=conf,
            entities=compact_entities,
            requires_clarification=clarify or chosen == "unknown",
            reason=reason,
            is_mock=True,
        )

    # Priority-ordered business intent rules
    if entities_obj.order_id and any(
        kw in lower for kw in ("where", "status", "track", "order", "arrive", "delivery", "when")
    ):
        return pick("order_status", 0.96, "Matched order ID and delivery/order status keywords.")

    if any(kw in lower for kw in ("where is my order", "order status", "track order", "track my order")):
        needs_id = entities_obj.order_id is None
        return pick(
            "order_status",
            0.91 if not needs_id else 0.84,
            "Matched order tracking intent.",
            clarify=needs_id,
        )

    if any(kw in lower for kw in ("refund", "money back", "return my", "return policy", "exchange")):
        return pick("refund_request", 0.93, "Matched refund/return keywords.")

    if any(kw in lower for kw in ("cancel", "cancellation")):
        return pick("cancellation_request", 0.91, "Matched cancellation keywords.")

    if any(kw in lower for kw in ("broken", "damaged", "complaint", "bad quality", "worst", "terrible", "defective")):
        return pick("complaint", 0.89, "Matched complaint/issue keywords.")

    if any(kw in lower for kw in ("appointment", "slot", "book a", "schedule", "counseling", "consultation")):
        return pick("appointment_booking", 0.92, "Matched appointment/consultation scheduling keywords.")

    if any(kw in lower for kw in ("hour", "open", "timing", "working days", "when are you open")):
        return pick("business_hours", 0.93, "Matched business hours keywords.")

    if any(kw in lower for kw in ("contact", "email", "phone number", "reach you", "support email")):
        return pick("contact_information", 0.92, "Matched contact information keywords.")

    if any(kw in lower for kw in ("course", "bootcamp", "syllabus", "duration", "prerequisites", "certificate", "enrollment", "cohort")):
        return pick("course_information", 0.93, "Matched course/education keywords.")

    if any(kw in lower for kw in ("cheapest", "cost", "price", "how much", "fee", "fees")) and not any(
        kw in lower for kw in ("find", "show", "search")
    ):
        return pick("pricing_inquiry", 0.90, "Matched pricing/cost inquiry keywords.")

    if any(
        kw in lower
        for kw in (
            "find",
            "show",
            "search",
            "hoodie",
            "tee",
            "t-shirt",
            "cargo",
            "product",
            "catalog",
            "wear",
            "clothing",
            "under",
        )
    ):
        return pick("product_inquiry", 0.94, "Matched product discovery/catalog search keywords.")

    if any(kw in lower for kw in ("shipping", "delivery", "payment", "cod", "upi", "policy", "faq", "services", "hello", "hi", "help")):
        return pick("general_inquiry", 0.82, "Matched general business policy/FAQ/greeting keywords.")

    return IntentResult(
        intent="unknown",
        confidence=0.35,
        entities=compact_entities,
        requires_clarification=True,
        reason="No confident business intent matched; clarification requested.",
        is_mock=True,
    )


def classify_intent(
    message: str,
    backend: GeminiAgentBackend | None = None,
    context: BusinessContext | None = None,
    allowed_intents: list[str] | tuple[str, ...] | None = None,
    recent_history: list[str] | None = None,
) -> IntentResult:
    """Task 1: Classify a user query into a structured business intent."""
    cleaned = validate_non_empty_text(message, "message")
    intents = _resolve_allowed_intents(context, allowed_intents)

    if backend is None or backend.is_mock:
        return _classify_deterministic(
            cleaned,
            allowed_intents=intents,
            context=context,
            recent_history=recent_history,
        )

    extracted = extract_structured_entities(
        cleaned,
        backend=backend,
        context=context,
        recent_history=recent_history,
    )
    compact_entities = extracted.to_dict(omit_none=True)

    business_info = (
        f"Active business: {context.business_name} ({context.industry})."
        if context is not None
        else "Active business: Generic."
    )
    history_block = (
        "\nRecent conversation context:\n" + "\n".join(f"- {h}" for h in recent_history)
        if recent_history
        else ""
    )

    system_instruction = (
        "You are a business intent classifier. Classify the user message into EXACTLY ONE "
        f"intent from this allowed list: {', '.join(intents)}.\n"
        f"{business_info}\n"
        "Return ONLY valid JSON with keys:\n"
        '{"intent": "<one_allowed_intent>", "confidence": <float 0.0-1.0>, '
        '"requires_clarification": <true|false>, "reason": "<short explanation>"}\n'
        "If the query is ambiguous, off-topic, or unintelligible, use intent='unknown' and "
        "requires_clarification=true."
    )

    result = backend.generate(
        user_prompt=f"User message: {cleaned}{history_block}",
        system_instruction=system_instruction,
        temperature=0.0,
        max_output_tokens=200,
    )

    parsed = _extract_json_object(result.text)
    if parsed is None:
        fallback = _classify_deterministic(
            cleaned,
            allowed_intents=intents,
            context=context,
            recent_history=recent_history,
        )
        return IntentResult(
            intent=fallback.intent,
            confidence=fallback.confidence,
            entities=compact_entities or fallback.entities,
            requires_clarification=fallback.requires_clarification,
            reason=f"Fallback classification (unparseable model JSON): {fallback.reason}",
            is_mock=False,
        )

    raw_intent = str(parsed.get("intent", "unknown")).strip()
    if raw_intent not in intents:
        raw_intent = "unknown"

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    requires_clarification = bool(parsed.get("requires_clarification", raw_intent == "unknown"))
    reason = str(parsed.get("reason", "Classified via Gemini API.")).strip()

    return IntentResult(
        intent=raw_intent,
        confidence=confidence,
        entities=compact_entities,
        requires_clarification=requires_clarification,
        reason=reason,
        is_mock=False,
    )
