import logging
from typing import Any

from supabase_client import get_exception_status_code, get_supabase_client

logger = logging.getLogger(__name__)


class SupabaseRateLimitError(Exception):
    """Raised when Supabase rejects an operation because of rate limiting."""


def create_auth_user_profile(
    email: str,
    password: str,
    first_name: str,
    last_name: str,
    assistant_name: str,
) -> dict[str, Any]:
    auth_client = get_supabase_client(use_service_role=False)
    try:
        signup_response = auth_client.auth.sign_up(
            {"email": email, "password": password}
        )
    except Exception as exc:
        if _is_rate_limit_error(exc):
            raise SupabaseRateLimitError from exc
        raise

    auth_user = signup_response.user

    if auth_user is None or not auth_user.id:
        raise ValueError("Supabase Auth did not return a user.")

    user_id = auth_user.id
    profile = {
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "assistant_name": assistant_name,
    }

    try:
        profile_response = (
            get_supabase_client(use_service_role=True)
            .table("users")
            .insert(profile)
            .select("user_id")
            .execute()
        )
    except Exception as exc:
        _remove_auth_user(user_id)
        if _is_rate_limit_error(exc):
            raise SupabaseRateLimitError from exc
        raise RuntimeError("Could not create the application user profile.") from exc

    profile_rows = profile_response.data or []
    if not profile_rows:
        _remove_auth_user(user_id)
        raise RuntimeError("Could not create the application user profile.")

    return {
        "user_id": user_id,
        "email": auth_user.email or email,
        "email_confirmation_required": signup_response.session is None,
    }


def create_admin_test_user_profile(
    email: str,
    password: str,
    first_name: str,
    last_name: str,
    assistant_name: str,
) -> dict[str, Any]:
    """Create a confirmed development test user and its matching profile."""
    admin_client = get_supabase_client(use_service_role=True)
    try:
        create_response = admin_client.auth.admin.create_user(
            {
                "email": email,
                "password": password,
                "email_confirm": True,
            }
        )
    except Exception as exc:
        if _is_rate_limit_error(exc):
            raise SupabaseRateLimitError from exc
        raise

    auth_user = create_response.user
    if auth_user is None or not auth_user.id:
        raise ValueError("Supabase Auth did not return a user.")

    profile = {
        "user_id": auth_user.id,
        "first_name": first_name,
        "last_name": last_name,
        "assistant_name": assistant_name,
    }

    try:
        profile_response = (
            admin_client.table("users")
            .insert(profile)
            .select("user_id")
            .execute()
        )
    except Exception as exc:
        if _is_rate_limit_error(exc):
            raise SupabaseRateLimitError from exc
        raise RuntimeError("Could not create the application user profile.") from exc

    if not profile_response.data:
        raise RuntimeError("Could not create the application user profile.")

    return {
        "user_id": auth_user.id,
        "email": auth_user.email or email,
        "email_confirmation_required": False,
    }


def _remove_auth_user(user_id: str) -> None:
    try:
        get_supabase_client(use_service_role=True).auth.admin.delete_user(user_id)
    except Exception:
        logger.exception("Failed to roll back Auth user %s after profile creation failure", user_id)


def _is_rate_limit_error(exc: BaseException) -> bool:
    return get_exception_status_code(exc) == 429
