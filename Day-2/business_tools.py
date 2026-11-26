"""Task 3: Reusable Business Tool System and Explicit Tool Registry for Day 2.

Provides a common BusinessTool interface and an allowlist-based ToolRegistry
so tools can be added without modifying the core agent. Tools validate all
inputs, never execute arbitrary code, and never perform irreversible actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import re
import time
from typing import Any

from business_config import (
    AgentError,
    BusinessContext,
    ValidationError,
    parse_price_inr,
    validate_non_empty_text,
)


class ToolRegistryError(AgentError):
    """Raised when an unapproved or unknown tool name is requested."""


@dataclass(frozen=True)
class ToolResult:
    """Structured result returned by every tool execution."""

    tool_name: str
    status: str  # "success", "not_found", "validation_error", "error"
    data: dict[str, Any]
    message: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BusinessTool(ABC):
    """Common interface for all Day 2 business tools."""

    name: str = ""
    description: str = ""
    required_args: tuple[str, ...] = ()

    @abstractmethod
    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize arguments before execution."""

    @abstractmethod
    def execute(self, args: dict[str, Any], context: BusinessContext) -> dict[str, Any]:
        """Run the deterministic tool action and return structured payload data."""


def _normalize_search_tokens(text: str) -> list[str]:
    """Tokenize search text and normalize simple English plurals (hoodies -> hoodie)."""
    tokens: list[str] = []
    for raw in re.split(r"[^a-z0-9]+", text.lower()):
        if len(raw) < 2:
            continue
        if raw.endswith("ies") and len(raw) > 4:
            tokens.append(raw[:-3] + "y")
        elif raw.endswith("s") and len(raw) > 3 and not raw.endswith("ss"):
            tokens.append(raw[:-1])
        else:
            tokens.append(raw)
    return [t for t in tokens if t not in {"find", "show", "under", "below", "with", "the", "for", "and", "inr"}]


class ProductSearchTool(BusinessTool):
    """Tool A: Search products or courses in the configured business catalog."""

    name = "product_search"
    description = "Search items/courses in the active business catalog with optional category or max_price filter."
    required_args = ()

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValidationError("Tool arguments must be a dictionary.")

        query = args.get("query")
        if query is not None:
            if not isinstance(query, str):
                raise ValidationError("'query' must be a string.")
            query = query.strip()

        category = args.get("category")
        if category is not None:
            if not isinstance(category, str):
                raise ValidationError("'category' must be a string.")
            category = category.strip()

        max_price = args.get("max_price")
        parsed_price: float | None = None
        if max_price is not None:
            if isinstance(max_price, bool):
                raise ValidationError("'max_price' cannot be a boolean.")
            parsed_price = parse_price_inr(max_price)
            if parsed_price is None or parsed_price < 0:
                raise ValidationError("'max_price' must be a non-negative number.")

        sort_by = args.get("sort_by")
        if sort_by is not None and sort_by not in {"price_asc", "price_desc"}:
            raise ValidationError("'sort_by' must be 'price_asc' or 'price_desc'.")

        return {
            "query": query or "",
            "category": category or "",
            "max_price": parsed_price,
            "sort_by": sort_by,
        }

    def execute(self, args: dict[str, Any], context: BusinessContext) -> dict[str, Any]:
        clean = self.validate_args(args)
        query_tokens = _normalize_search_tokens(clean["query"])
        category_filter = clean["category"].lower()
        max_price = clean["max_price"]

        matches: list[dict[str, Any]] = []
        for item in context.catalog:
            searchable = " ".join(
                str(v) for k, v in item.items() if k != "numeric_price"
            ).lower()
            item_category = str(item.get("category", "")).lower()

            if category_filter and category_filter not in item_category:
                if not any(tok in searchable for tok in _normalize_search_tokens(category_filter)):
                    continue

            if query_tokens and not all(tok in searchable for tok in query_tokens):
                continue

            item_price = item.get("numeric_price")
            if max_price is not None:
                if not isinstance(item_price, (int, float)) or item_price > max_price:
                    continue

            display_item = {k: v for k, v in item.items() if k != "numeric_price"}
            if isinstance(item_price, (int, float)):
                display_item["_numeric_price"] = item_price
            matches.append(display_item)

        if clean["sort_by"] == "price_asc":
            matches.sort(key=lambda x: x.get("_numeric_price", float("inf")))
        elif clean["sort_by"] == "price_desc":
            matches.sort(key=lambda x: x.get("_numeric_price", -1.0), reverse=True)

        for item in matches:
            item.pop("_numeric_price", None)

        return {
            "business": context.business_name,
            "filters": {
                "query": clean["query"],
                "category": clean["category"],
                "max_price": max_price,
                "sort_by": clean["sort_by"],
            },
            "count": len(matches),
            "items": matches,
        }


class BusinessInfoTool(BusinessTool):
    """Tool B: Return business hours, contact details, services, and official policies."""

    name = "business_info"
    description = "Retrieve verified business hours, contact details, services, and official policies."
    required_args = ()

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValidationError("Tool arguments must be a dictionary.")
        topic = args.get("topic")
        if topic is not None:
            if not isinstance(topic, str):
                raise ValidationError("'topic' must be a string.")
            topic = topic.strip().lower()
        return {"topic": topic or "all"}

    def execute(self, args: dict[str, Any], context: BusinessContext) -> dict[str, Any]:
        clean = self.validate_args(args)
        topic = clean["topic"]

        payload: dict[str, Any] = {
            "business": context.business_name,
            "industry": context.industry,
            "support_hours": context.support_hours,
            "contact": context.contact,
            "services": context.profile.services,
        }

        if topic in {"refund", "return", "exchange"}:
            payload["policies"] = {
                k: v for k, v in context.policies.items()
                if any(word in k.lower() for word in ("refund", "return", "exchange"))
            }
        elif topic in {"shipping", "delivery"}:
            payload["policies"] = {
                k: v for k, v in context.policies.items()
                if any(word in k.lower() for word in ("ship", "delivery", "track"))
            }
        elif topic in {"hours", "contact"}:
            payload["policies"] = {}
        else:
            payload["policies"] = context.policies

        return payload


class OrderLookupTool(BusinessTool):
    """Tool C: Look up order or enrollment status using mock business order data."""

    name = "order_lookup"
    description = "Look up an order or enrollment status by required order_id (mock data only)."
    required_args = ("order_id",)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValidationError("Tool arguments must be a dictionary.")
        raw_id = args.get("order_id")
        cleaned = validate_non_empty_text(raw_id, "order_id").upper()
        if not re.match(r"^[A-Z0-9-]{3,20}$", cleaned):
            raise ValidationError(
                "'order_id' must be 3-20 alphanumeric/hyphen characters (e.g., DZ1024)."
            )
        return {"order_id": cleaned}

    def execute(self, args: dict[str, Any], context: BusinessContext) -> dict[str, Any]:
        clean = self.validate_args(args)
        order_id = clean["order_id"]
        record = context.orders.get(order_id)

        if record is None:
            return {
                "found": False,
                "is_mock_data": True,
                "order_id": order_id,
                "business": context.business_name,
                "message": f"No order record found for '{order_id}' in {context.business_name} mock records.",
            }

        return {
            "found": True,
            "is_mock_data": True,
            "order_id": order_id,
            "business": context.business_name,
            "order": record,
        }


class AppointmentAvailabilityTool(BusinessTool):
    """Tool D: Check available consultation/appointment slots without booking."""

    name = "appointment_availability"
    description = "List available appointment or counseling slots (read-only availability check)."
    required_args = ()

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValidationError("Tool arguments must be a dictionary.")
        date_filter = args.get("date")
        if date_filter is not None:
            if not isinstance(date_filter, str):
                raise ValidationError("'date' must be a string.")
            date_filter = date_filter.strip()
        topic = args.get("topic")
        if topic is not None:
            if not isinstance(topic, str):
                raise ValidationError("'topic' must be a string.")
            topic = topic.strip().lower()
        return {"date": date_filter or "", "topic": topic or ""}

    def execute(self, args: dict[str, Any], context: BusinessContext) -> dict[str, Any]:
        clean = self.validate_args(args)
        available_slots = [
            slot for slot in context.appointment_slots if bool(slot.get("available"))
        ]

        if clean["date"] and re.match(r"^\d{4}-\d{2}-\d{2}$", clean["date"]):
            available_slots = [
                slot for slot in available_slots if slot.get("date") == clean["date"]
            ]

        if clean["topic"]:
            topic_tokens = _normalize_search_tokens(clean["topic"])
            filtered = [
                slot for slot in available_slots
                if any(tok in str(slot.get("topic", "")).lower() for tok in topic_tokens)
            ]
            if filtered:
                available_slots = filtered

        return {
            "business": context.business_name,
            "booking_performed": False,
            "is_mock_data": True,
            "available_count": len(available_slots),
            "slots": available_slots,
            "note": "Availability preview only. No appointment has been booked.",
        }


class KnowledgeBaseSearchTool(BusinessTool):
    """Tool E: Search configured FAQs, policies, and services for grounded facts."""

    name = "knowledge_base_search"
    description = "Search business FAQs, policies, and service descriptions."
    required_args = ("query",)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValidationError("Tool arguments must be a dictionary.")
        query = validate_non_empty_text(args.get("query"), "query")
        return {"query": query}

    def execute(self, args: dict[str, Any], context: BusinessContext) -> dict[str, Any]:
        clean = self.validate_args(args)
        tokens = _normalize_search_tokens(clean["query"])

        matched_faqs: list[dict[str, Any]] = []
        for faq in context.faqs:
            haystack = " ".join(
                [
                    str(faq.get("question", "")),
                    str(faq.get("answer", "")),
                    " ".join(str(k) for k in faq.get("keywords", [])),
                ]
            ).lower()
            if not tokens or any(tok in haystack for tok in tokens):
                matched_faqs.append(
                    {"question": faq.get("question"), "answer": faq.get("answer")}
                )

        matched_policies: dict[str, str] = {}
        for key, text in context.policies.items():
            haystack = f"{key.replace('_', ' ')} {text}".lower()
            if not tokens or any(tok in haystack for tok in tokens):
                matched_policies[key] = text

        return {
            "business": context.business_name,
            "query": clean["query"],
            "faqs": matched_faqs,
            "policies": matched_policies,
            "services": context.profile.services,
        }


class ToolRegistry:
    """Explicit allowlist registry governing which tools the agent may execute."""

    def __init__(self) -> None:
        self._tools: dict[str, BusinessTool] = {}

    def register(self, tool: BusinessTool) -> None:
        if not isinstance(tool, BusinessTool):
            raise ValidationError("Only BusinessTool instances can be registered.")
        name = validate_non_empty_text(tool.name, "tool.name")
        self._tools[name] = tool

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get(self, tool_name: str) -> BusinessTool:
        if tool_name not in self._tools:
            raise ToolRegistryError(
                f"Tool '{tool_name}' is not in the approved ToolRegistry allowlist."
            )
        return self._tools[tool_name]

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "required_args": list(tool.required_args),
            }
            for tool in self._tools.values()
        ]

    def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: BusinessContext,
    ) -> ToolResult:
        """Validate tool allowance and arguments, run tool, and measure duration."""
        start = time.perf_counter()
        try:
            tool = self.get(tool_name)
            validated_args = tool.validate_args(args)
            data = tool.execute(validated_args, context)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)

            if tool_name == "order_lookup" and data.get("found") is False:
                return ToolResult(
                    tool_name=tool_name,
                    status="not_found",
                    data=data,
                    message=str(data.get("message", "Order not found.")),
                    duration_ms=duration_ms,
                )

            return ToolResult(
                tool_name=tool_name,
                status="success",
                data=data,
                message=f"Executed '{tool_name}' successfully.",
                duration_ms=duration_ms,
            )
        except ValidationError as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            return ToolResult(
                tool_name=tool_name,
                status="validation_error",
                data={},
                message=str(exc),
                duration_ms=duration_ms,
            )
        except ToolRegistryError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch for unexpected errors
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            return ToolResult(
                tool_name=tool_name,
                status="error",
                data={},
                message=f"Tool execution failed: {exc}",
                duration_ms=duration_ms,
            )


def create_default_tool_registry() -> ToolRegistry:
    """Create the default Day 2 ToolRegistry containing all five approved tools."""
    registry = ToolRegistry()
    registry.register(ProductSearchTool())
    registry.register(BusinessInfoTool())
    registry.register(OrderLookupTool())
    registry.register(AppointmentAvailabilityTool())
    registry.register(KnowledgeBaseSearchTool())
    return registry
