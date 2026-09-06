"""
profile_updater.py — Background entity extraction for user business profiles.

Every 5 chat turns, this module calls Gemini to analyse the recent
conversation and extract structured business facts, then upserts
them into the user_profiles table.
"""

import json
from datetime import datetime, timezone
import logging
import re
from typing import Optional

# pyrefly: ignore [missing-import]
import google.generativeai as genai
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from database import (
    UserProfile,
    get_or_create_profile,
    get_recent_messages_for_extraction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity-extraction system prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a precise data extraction assistant.
Your task is to analyse a conversation between a rural micro-entrepreneur and a business advisor AI,
then extract or update structured facts about the entrepreneur.

Return ONLY a valid JSON object (no markdown, no explanation) with these keys
(use null for any field you cannot determine, do NOT guess):

{
  "name":             string or null,
  "business_type":    string or null,
  "location":         string or null,
  "turnover":         string or null,
  "employee_count":   string or null,
  "schemes_enrolled": string or null,
  "goals":            string or null,
  "challenges":       string or null,
  "extra_facts":      { "key": "value", ... } or {}
}

Guidelines:
- "schemes_enrolled" should be a comma-separated list of government scheme names mentioned.
- "extra_facts" should capture any important business details that don't fit other fields
  (e.g., "new_store_opened": "October 2024", "mudra_loan_applied": "yes").
- Only include facts that are EXPLICITLY stated in the conversation. Do NOT infer.
- If the conversation contains no new factual information, return all null values.
"""


def _call_gemini_for_extraction(conversation_text: str) -> Optional[dict]:
    """
    Call Gemini (gemini-3.6-flash) to extract structured entity facts
    from a conversation snippet.
    Returns a dict on success, None on failure.
    """
    prompt = f"""Here is a recent conversation to analyse:

{conversation_text}

Extract the structured business profile facts as instructed."""

    raw = ""  # Initialised here so it is always bound in the except handlers
    try:
        model = genai.GenerativeModel(
            model_name="gemini-3.6-flash",
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        raw = response.text.strip()

        # Strip markdown code fences if Gemini wraps it anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        extracted = json.loads(raw)
        logger.debug(f"Extraction result: {extracted}")
        return extracted

    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse error in extraction: {exc} | raw={raw!r}")
        return None
    except Exception as exc:
        logger.error(f"Gemini entity extraction failed: {exc}")
        return None


def _merge_profile(profile: UserProfile, extracted: dict) -> bool:
    """
    Merge extracted facts into the UserProfile object.
    Returns True if any field was actually updated.
    """
    changed = False

    simple_fields = [
        "name", "business_type", "location",
        "turnover", "employee_count", "goals", "challenges",
    ]
    for field in simple_fields:
        value = extracted.get(field)
        if value and value != getattr(profile, field):
            setattr(profile, field, value)
            changed = True

    # schemes_enrolled: merge/append new schemes
    new_schemes = extracted.get("schemes_enrolled")
    if new_schemes:
        existing = set(
            s.strip()
            for s in (profile.schemes_enrolled or "").split(",")
            if s.strip()
        )
        for scheme in new_schemes.split(","):
            scheme = scheme.strip()
            if scheme:
                existing.add(scheme)
        merged = ", ".join(sorted(existing))
        if merged != profile.schemes_enrolled:
            profile.schemes_enrolled = merged
            changed = True

    # extra_facts: merge JSON objects
    extra_new = extracted.get("extra_facts", {})
    if extra_new and isinstance(extra_new, dict):
        existing_extra: dict = {}
        if profile.extra_facts_json:
            try:
                existing_extra = json.loads(profile.extra_facts_json)
            except Exception:
                pass
        existing_extra.update(extra_new)
        new_json = json.dumps(existing_extra, ensure_ascii=False)
        if new_json != profile.extra_facts_json:
            profile.extra_facts_json = new_json
            changed = True

    return changed


def run_profile_update(db: Session, user_id: str) -> bool:
    """
    Main entry point called every 5 turns.

    1. Fetches recent messages from DB.
    2. Calls Gemini to extract structured facts.
    3. Merges & persists the updated profile.

    Returns True if the profile was actually updated.
    """
    try:
        messages = get_recent_messages_for_extraction(db, user_id, limit=10)
        if not messages:
            return False

        # Build conversation text
        lines = []
        for msg in messages:
            role = "Entrepreneur" if msg.role == "user" else "Advisor AI"
            lines.append(f"{role}: {msg.content}")
        conversation_text = "\n".join(lines)

        extracted = _call_gemini_for_extraction(conversation_text)
        if not extracted:
            return False

        profile = get_or_create_profile(db, user_id)
        changed = _merge_profile(profile, extracted)

        if changed:
            profile.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(profile)
            logger.info(f"Profile updated for user {user_id}")

        return changed

    except Exception as exc:
        logger.error(f"Profile update failed for user {user_id}: {exc}")
        db.rollback()
        return False


def should_run_profile_update(turn_count: int, update_interval: int = 5) -> bool:
    """Return True if this turn should trigger a profile update."""
    return turn_count > 0 and turn_count % update_interval == 0
