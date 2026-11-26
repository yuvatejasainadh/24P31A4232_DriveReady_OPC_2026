"""Day 2 configuration bridge reusing Day 1 profiles and Gemini backend.

Loads Day-1/task.py under a distinct module name (`day1_task`) so it never
collides with Day-2/task.py, and merges Day-1/business_profiles.json with
Day-2/data/business_facts.json without duplicating Day 1 profile definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DAY1_TASK_PATH = REPO_ROOT / "Day-1" / "task.py"
DEFAULT_DAY1_PROFILES_PATH = REPO_ROOT / "Day-1" / "business_profiles.json"
DEFAULT_DAY2_FACTS_PATH = Path(__file__).resolve().parent / "data" / "business_facts.json"


def _load_day1_module() -> Any:
    """Import Day-1/task.py safely as `day1_task` without shadowing Day-2/task.py."""
    existing = sys.modules.get("day1_task")
    if existing is not None:
        return existing

    if not DAY1_TASK_PATH.exists():
        raise FileNotFoundError(f"Required Day 1 module not found at {DAY1_TASK_PATH}")

    spec = importlib.util.spec_from_file_location("day1_task", DAY1_TASK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {DAY1_TASK_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["day1_task"] = module
    spec.loader.exec_module(module)
    return module


day1_task = _load_day1_module()

AgentError = day1_task.AgentError
ConfigurationError = day1_task.ConfigurationError
ValidationError = day1_task.ValidationError
LLMExecutionError = day1_task.LLMExecutionError
BusinessProfile = day1_task.BusinessProfile
ConversationTurn = day1_task.ConversationTurn
GeminiAgentBackend = day1_task.GeminiAgentBackend
load_environment = day1_task.load_environment
validate_non_empty_text = day1_task.validate_non_empty_text


@dataclass(frozen=True)
class BusinessContext:
    """Unified runtime configuration combining a Day 1 profile and Day 2 facts."""

    profile: BusinessProfile
    catalog: list[dict[str, Any]]
    policies: dict[str, str]
    faqs: list[dict[str, Any]]
    orders: dict[str, dict[str, Any]]
    appointment_slots: list[dict[str, Any]]
    supported_intents: list[str]
    support_hours: str
    contact: str

    @property
    def business_id(self) -> str:
        return self.profile.business_id

    @property
    def business_name(self) -> str:
        return self.profile.business_name

    @property
    def industry(self) -> str:
        return self.profile.industry


def parse_price_inr(value: Any) -> float | None:
    """Parse an INR price string such as 'INR 1,799' into a float (1799.0)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)", value)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def load_business_contexts(
    profiles_path: Path = DEFAULT_DAY1_PROFILES_PATH,
    facts_path: Path = DEFAULT_DAY2_FACTS_PATH,
) -> tuple[dict[str, BusinessContext], str]:
    """Load Day 1 business profiles and enrich them with Day 2 operational facts."""
    profiles, default_id = day1_task.load_business_profiles(profiles_path)

    raw_facts: dict[str, Any] = {}
    if facts_path.exists():
        try:
            loaded = json.loads(facts_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Malformed JSON in Day 2 business facts '{facts_path}': {exc}"
            ) from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("businesses", {}), dict):
            raise ConfigurationError(
                "Day 2 business_facts.json must contain a 'businesses' object."
            )
        raw_facts = loaded["businesses"]

    contexts: dict[str, BusinessContext] = {}
    for business_id, profile in profiles.items():
        kb = profile.knowledge_base
        facts = raw_facts.get(business_id, {})
        if not isinstance(facts, dict):
            raise ConfigurationError(f"Facts entry for '{business_id}' must be a JSON object.")

        base_catalog = [dict(item) for item in kb.get("catalog", []) if isinstance(item, dict)]
        extra_catalog = [
            dict(item) for item in facts.get("additional_catalog", []) if isinstance(item, dict)
        ]
        merged_catalog = [*base_catalog, *extra_catalog]
        for item in merged_catalog:
            if "numeric_price" not in item:
                parsed = parse_price_inr(item.get("price"))
                if parsed is not None:
                    item["numeric_price"] = parsed

        policies = {
            str(k): str(v)
            for k, v in kb.get("policies", {}).items()
            if isinstance(kb.get("policies"), dict)
        }
        faqs = [dict(f) for f in facts.get("faqs", []) if isinstance(f, dict)]
        raw_orders = facts.get("orders", {})
        orders = (
            {str(k).upper(): dict(v) for k, v in raw_orders.items() if isinstance(v, dict)}
            if isinstance(raw_orders, dict)
            else {}
        )
        slots = [
            dict(s) for s in facts.get("appointment_slots", []) if isinstance(s, dict)
        ]
        supported_intents = [
            str(intent).strip()
            for intent in facts.get("supported_intents", [])
            if isinstance(intent, str) and intent.strip()
        ]

        contexts[business_id] = BusinessContext(
            profile=profile,
            catalog=merged_catalog,
            policies=policies,
            faqs=faqs,
            orders=orders,
            appointment_slots=slots,
            supported_intents=supported_intents,
            support_hours=str(kb.get("support_hours", "Not specified")),
            contact=str(kb.get("contact", "Not specified")),
        )

    return contexts, default_id
