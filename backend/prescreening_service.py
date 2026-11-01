import asyncio
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_PHONE_DIGITS = re.compile(r"\D+")


class DatabaseQueryError(RuntimeError):
    """Raised when a Supabase database query fails."""


class PreScreeningContext(BaseModel):
    caller_phone_number: str
    normalized_caller_phone_number: str
    contacts: list[dict[str, Any]] = Field(default_factory=list)
    priority_contacts: list[dict[str, Any]] = Field(default_factory=list)
    spam_cache: list[dict[str, Any]] = Field(default_factory=list)
    call_history: list[dict[str, Any]] = Field(default_factory=list)
    user_preferences: list[dict[str, Any]] = Field(default_factory=list)


def normalize_phone_number(phone_number: str) -> str:
    normalized = _PHONE_DIGITS.sub("", phone_number)
    if not normalized:
        raise ValueError("Caller phone number must contain digits.")
    return normalized


def _phone_matches(phone_number: Any, normalized_phone_number: str) -> bool:
    if not isinstance(phone_number, str):
        return False
    try:
        stored_norm = normalize_phone_number(phone_number)
    except ValueError:
        return False

    if stored_norm == normalized_phone_number:
        return True

    # If both numbers have at least 10 digits (standard national number length),
    # match if one ends with the other (to handle country code prefixes like +91 or trunk 0).
    if min(len(stored_norm), len(normalized_phone_number)) >= 10:
        return (
            stored_norm.endswith(normalized_phone_number)
            or normalized_phone_number.endswith(stored_norm)
        )

    return False


def _check_user_profile_exists(client: Any, user_id: str) -> bool:
    """Verify that the authenticated user profile exists in public.users."""
    try:
        response = (
            client.table("users")
            .select("user_id")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return bool(response.data)
    except Exception as exc:
        logger.warning(
            "Could not verify public.users profile for user %s: %s",
            user_id,
            exc,
        )
        return False


def _fetch_all_prescreening_data(
    client: Any,
    user_id: str,
    normalized_phone_number: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Fetch all prescreening data sequentially on a worker thread to prevent socket concurrency issues."""
    contacts = _query_contacts(client, user_id)
    priority_contacts = _query_priority_contacts(client, user_id)
    spam_cache = _query_spam_cache(client, normalized_phone_number)
    call_history = _query_call_history(client, user_id)
    user_preferences = _query_user_preferences(client, user_id)
    return contacts, priority_contacts, spam_cache, call_history, user_preferences


async def get_prescreening_context(
    user_id: str,
    caller_phone_number: str,
    client: Any = None,
) -> PreScreeningContext:
    normalized_phone_number = normalize_phone_number(caller_phone_number)
    if client is None:
        client = get_supabase_client(use_service_role=True)

    # Diagnostic check: warn if profile is missing in public.users
    if not _check_user_profile_exists(client, user_id):
        logger.warning(
            "Authenticated user %s does not have an active profile in public.users",
            user_id,
        )

    responses = await asyncio.to_thread(
        _fetch_all_prescreening_data,
        client,
        user_id,
        normalized_phone_number,
    )

    contacts, priority_contacts, spam_cache, call_history, user_preferences = responses

    return PreScreeningContext(
        caller_phone_number=caller_phone_number,
        normalized_caller_phone_number=normalized_phone_number,
        contacts=[
            row
            for row in contacts
            if _phone_matches(row.get("phone_number"), normalized_phone_number)
        ],
        priority_contacts=[
            row
            for row in priority_contacts
            if _phone_matches(row.get("phone_number"), normalized_phone_number)
        ],
        spam_cache=spam_cache,
        call_history=[
            row
            for row in call_history
            if _phone_matches(row.get("caller_number"), normalized_phone_number)
        ],
        user_preferences=user_preferences,
    )


def _query_contacts(client: Any, user_id: str) -> list[dict[str, Any]]:
    try:
        response = (
            client.table("contacts")
            .select(
                "contact_id,user_id,contact_name,phone_number,"
                "created_at,updated_at,is_active"
            )
            .eq("user_id", user_id)
            .eq("is_active", True)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.exception("Failed to query contacts for user %s: %s", user_id, exc)
        raise DatabaseQueryError(f"Failed to query contacts: {exc}") from exc


def _query_priority_contacts(client: Any, user_id: str) -> list[dict[str, Any]]:
    try:
        response = (
            client.table("priority_contacts")
            .select(
                "priority_contact_id,user_id,contact_name,phone_number,"
                "category,is_active,created_at,updated_at"
            )
            .eq("user_id", user_id)
            .eq("is_active", True)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.exception("Failed to query priority contacts for user %s: %s", user_id, exc)
        raise DatabaseQueryError(f"Failed to query priority contacts: {exc}") from exc


def _query_spam_cache(
    client: Any,
    normalized_phone_number: str,
) -> list[dict[str, Any]]:
    try:
        response = (
            client.table("spam_cache")
            .select("spam_id,phone_number,source,created_at,updated_at,is_active")
            .eq("is_active", True)
            .execute()
        )
        return [
            row
            for row in (response.data or [])
            if _phone_matches(row.get("phone_number"), normalized_phone_number)
        ]
    except Exception as exc:
        logger.exception("Failed to query spam cache: %s", exc)
        raise DatabaseQueryError(f"Failed to query spam cache: {exc}") from exc


def _query_call_history(client: Any, user_id: str) -> list[dict[str, Any]]:
    try:
        response = (
            client.table("call_history")
            .select(
                "call_id,user_id,caller_number,call_started_at,call_ended_at,"
                "caller_type,classification,decision,outcome,created_at"
            )
            .eq("user_id", user_id)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.exception("Failed to query call history for user %s: %s", user_id, exc)
        raise DatabaseQueryError(f"Failed to query call history: {exc}") from exc


def _query_user_preferences(client: Any, user_id: str) -> list[dict[str, Any]]:
    try:
        response = (
            client.table("user_preferences")
            .select(
                "preference_id,user_id,preference_text,source,confidence,"
                "is_active,created_at,updated_at"
            )
            .eq("user_id", user_id)
            .eq("is_active", True)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.exception("Failed to query user preferences for user %s: %s", user_id, exc)
        raise DatabaseQueryError(f"Failed to query user preferences: {exc}") from exc
