"""Task 2: Structured Information Extraction for Day 2.

Extracts normalized business entities from natural-language user messages using
Gemini in live mode and deterministic pattern matching in demo mode. Missing
fields remain None and are never invented.
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
    parse_price_inr,
    validate_non_empty_text,
)


ENTITY_FIELDS = (
    "customer_name",
    "email",
    "phone_number",
    "order_id",
    "product_name",
    "product_category",
    "price",
    "date",
    "appointment_time",
    "course_name",
    "location",
    "requested_action",
)


@dataclass(frozen=True)
class ExtractedEntities:
    """Validated structured entities extracted from a user message."""

    customer_name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    order_id: str | None = None
    product_name: str | None = None
    product_category: str | None = None
    price: float | None = None
    date: str | None = None
    appointment_time: str | None = None
    course_name: str | None = None
    location: str | None = None
    requested_action: str | None = None

    def to_dict(self, omit_none: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable dict, optionally omitting missing fields."""
        data = asdict(self)
        if omit_none:
            return {k: v for k, v in data.items() if v is not None}
        return data


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Safely extract a JSON object from raw LLM output."""
    cleaned = text.strip()
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


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    if not stripped or stripped.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    return stripped


def normalize_extracted_entities(raw: dict[str, Any]) -> ExtractedEntities:
    """Validate and normalize a raw entity dictionary into ExtractedEntities."""
    if not isinstance(raw, dict):
        raise ValidationError("Extracted entities payload must be a JSON object.")

    order_id = _clean_optional_str(raw.get("order_id"))
    if order_id is not None:
        order_id = re.sub(r"[^A-Za-z0-9-]", "", order_id).upper()
        if not re.search(r"\d", order_id):
            order_id = None

    email = _clean_optional_str(raw.get("email"))
    if email is not None and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        email = None

    phone = _clean_optional_str(raw.get("phone_number"))
    if phone is not None:
        digits = re.sub(r"\D", "", phone)
        if len(digits) < 7 or len(digits) > 15:
            phone = None

    raw_price = raw.get("price")
    price: float | None = None
    if raw_price is not None:
        parsed_price = parse_price_inr(raw_price)
        if parsed_price is not None and parsed_price >= 0:
            price = parsed_price

    return ExtractedEntities(
        customer_name=_clean_optional_str(raw.get("customer_name")),
        email=email,
        phone_number=phone,
        order_id=order_id,
        product_name=_clean_optional_str(raw.get("product_name")),
        product_category=_clean_optional_str(raw.get("product_category")),
        price=price,
        date=_clean_optional_str(raw.get("date")),
        appointment_time=_clean_optional_str(raw.get("appointment_time")),
        course_name=_clean_optional_str(raw.get("course_name")),
        location=_clean_optional_str(raw.get("location")),
        requested_action=_clean_optional_str(raw.get("requested_action")),
    )


def _infer_requested_action(lower_msg: str, order_id: str | None) -> str | None:
    """Map natural-language verbs/phrases to normalized action identifiers."""
    if any(kw in lower_msg for kw in ("delivery", "arrive", "when will", "track", "where is")):
        if order_id or "order" in lower_msg:
            return "check_delivery_status"
    if any(kw in lower_msg for kw in ("status", "check my order")) and (order_id or "order" in lower_msg):
        return "check_order_status"
    if any(kw in lower_msg for kw in ("refund", "money back")):
        return "request_refund"
    if any(kw in lower_msg for kw in ("cancel", "cancellation")):
        return "cancel_order"
    if any(kw in lower_msg for kw in ("book", "appointment", "slot", "schedule", "counseling")):
        return "check_appointment_availability"
    if any(kw in lower_msg for kw in ("find", "show", "search", "looking for", "cheapest", "under")):
        return "search_catalog"
    if any(kw in lower_msg for kw in ("hours", "open", "timing")):
        return "get_business_hours"
    if any(kw in lower_msg for kw in ("contact", "email", "phone", "reach")):
        return "get_contact_info"
    return None


def _extract_entities_deterministic(
    message: str,
    context: BusinessContext | None = None,
    recent_history: list[str] | None = None,
) -> ExtractedEntities:
    """Deterministic entity extraction for offline demo mode."""
    lower_msg = message.lower()

    # Order ID (e.g., DZ1024, ES2041, ORD-1002)
    order_match = re.search(r"\b([A-Z]{2,4}[-#]?\d{3,6})\b", message, flags=re.IGNORECASE)
    order_id = order_match.group(1).replace("#", "").upper() if order_match else None

    # Email
    email_match = re.search(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", message)
    email = email_match.group(1) if email_match else None

    # Phone number (10-15 digits optionally starting with +, excluding ISO dates like 2026-11-28)
    phone_match = re.search(r"(\+?\d[\d\s-]{8,14}\d)", message)
    phone = phone_match.group(1).strip() if phone_match else None
    if phone and (
        re.match(r"^\d{4}-\d{2}-\d{2}$", phone)
        or len(re.sub(r"\D", "", phone)) < 10
        or (order_id and order_id in phone)
    ):
        phone = None

    # Customer name (explicit introduction only)
    name_match = re.search(
        r"\b(?:my name is|this is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        message,
    )
    customer_name = name_match.group(1).strip() if name_match else None

    # Price constraint / mentioned price (e.g., "under ₹1000", "below 1500", "INR 899")
    price: float | None = None
    price_match = re.search(
        r"(?:under|below|less than|max|budget|around|rs\.?|inr|₹)\s*₹?\s*(\d[\d,]*(?:\.\d+)?)",
        lower_msg,
    )
    if price_match:
        price = parse_price_inr(price_match.group(1))

    # Date & Appointment Time
    date_match = re.search(
        r"\b(\d{4}-\d{2}-\d{2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lower_msg,
    )
    date_val = date_match.group(1) if date_match else None

    time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lower_msg)
    appointment_time = time_match.group(1).upper() if time_match else None

    # Location (e.g., "in Hyderabad", "in Mumbai")
    location_match = re.search(
        r"\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        message,
    )
    location = location_match.group(1) if location_match else None

    # Product / Category / Course matching
    product_name: str | None = None
    product_category: str | None = None
    course_name: str | None = None

    # Common color + garment phrase (e.g., "black hoodie", "black hoodies", "cargo trousers")
    phrase_match = re.search(
        r"\b((?:black|white|grey|oversized|heavyweight|zip|pullover|graphic|essential|relaxed)?\s*"
        r"(?:hoodies?|t-?shirts?|tees?|cargo\s+trousers?|trousers?))\b",
        lower_msg,
    )
    if phrase_match:
        candidate = " ".join(phrase_match.group(1).split())
        if candidate.endswith("hoodies"):
            candidate = candidate[:-1]
        elif candidate.endswith("tees"):
            candidate = candidate[:-1]
        product_name = candidate

    if context is not None:
        is_education = "education" in context.industry.lower() or "edtech" in context.industry.lower()
        for item in context.catalog:
            item_title = str(item.get("item", ""))
            category = str(item.get("category", ""))
            if item_title and item_title.lower() in lower_msg:
                if is_education:
                    course_name = item_title
                else:
                    product_name = item_title
            if category and any(tok in lower_msg for tok in category.lower().split() if len(tok) > 3):
                product_category = category

        if is_education and course_name is None:
            for item in context.catalog:
                title = str(item.get("item", ""))
                tokens = [t for t in re.split(r"[^a-z0-9]+", title.lower()) if len(t) > 3]
                if sum(1 for t in tokens if t in lower_msg) >= 2:
                    course_name = title
                    break

    if product_category is None and product_name:
        if "hoodie" in product_name:
            product_category = "Hoodies & Sweatshirts"
        elif "tee" in product_name or "shirt" in product_name:
            product_category = "T-Shirts"
        elif "cargo" in product_name or "trouser" in product_name:
            product_category = "Bottoms"

    # Follow-up resolution from recent session memory (e.g., "Which one is the cheapest?")
    if recent_history and not product_name and not course_name and not order_id:
        if any(tok in lower_msg for tok in ("which one", "cheapest", "lowest", "compare", "that", "it")):
            user_lines = [
                line[6:] if line.lower().startswith("user: ") else line
                for line in recent_history
                if not line.lower().startswith("assistant: ")
            ]
            prior_joined = " ".join(user_lines) if user_lines else " ".join(recent_history)
            prior_entities = _extract_entities_deterministic(prior_joined, context=context, recent_history=None)
            product_name = prior_entities.product_name
            product_category = prior_entities.product_category
            course_name = prior_entities.course_name

    requested_action = _infer_requested_action(lower_msg, order_id)

    return normalize_extracted_entities(
        {
            "customer_name": customer_name,
            "email": email,
            "phone_number": phone,
            "order_id": order_id,
            "product_name": product_name,
            "product_category": product_category,
            "price": price,
            "date": date_val,
            "appointment_time": appointment_time,
            "course_name": course_name,
            "location": location,
            "requested_action": requested_action,
        }
    )


def extract_structured_entities(
    message: str,
    backend: GeminiAgentBackend | None = None,
    context: BusinessContext | None = None,
    recent_history: list[str] | None = None,
) -> ExtractedEntities:
    """Extract and validate structured business entities from a user query."""
    cleaned = validate_non_empty_text(message, "message")

    if backend is None or backend.is_mock:
        return _extract_entities_deterministic(
            cleaned,
            context=context,
            recent_history=recent_history,
        )

    business_hint = (
        f"Business: {context.business_name} ({context.industry})"
        if context is not None
        else "Business: Generic"
    )
    history_block = (
        "\nRecent Conversation Context:\n" + "\n".join(f"- {h}" for h in recent_history)
        if recent_history
        else ""
    )

    system_instruction = (
        "You are a strict structured information extractor. Return ONLY a single valid JSON object "
        f"with keys: {', '.join(ENTITY_FIELDS)}.\n"
        "Rules:\n"
        "1. Use null for any field not explicitly stated in the user message (or direct follow-up context).\n"
        "2. NEVER invent customer names, emails, phone numbers, prices, dates, or order IDs.\n"
        "3. Return numeric values (without currency symbols) for 'price' when mentioned.\n"
        f"{business_hint}"
    )
    prompt = f"User Message: {cleaned}{history_block}"

    result = backend.generate(
        user_prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.0,
        max_output_tokens=300,
    )
    parsed = _extract_json_block(result.text)
    if parsed is None:
        return _extract_entities_deterministic(
            cleaned,
            context=context,
            recent_history=recent_history,
        )
    return normalize_extracted_entities(parsed)
