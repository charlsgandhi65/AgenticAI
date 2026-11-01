import os
from pathlib import Path

from dotenv import load_dotenv
import httpx
from supabase import Client, create_client

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)
load_dotenv(override=False)

LOCAL_DEVELOPMENT_ENVS = frozenset({"local", "development", "dev"})


def get_supabase_client(use_service_role: bool = False) -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    if use_service_role:
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        missing_key_name = "SUPABASE_SERVICE_ROLE_KEY"
    else:
        # Backward compatible fallback to SUPABASE_KEY.
        supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
        missing_key_name = "SUPABASE_ANON_KEY (or SUPABASE_KEY)"

    if not supabase_url:
        raise ValueError("SUPABASE_URL must be set in .env")
    if not supabase_key:
        raise ValueError(f"{missing_key_name} must be set in .env")

    client = create_client(supabase_url, supabase_key)

    # Increase gotrue auth client timeouts from default 5.0s to 30.0s to prevent transient ReadTimeout on cloud calls
    auth_timeout = httpx.Timeout(30.0)
    auth_obj = getattr(client, "auth", None)
    if auth_obj is not None:
        http_client = getattr(auth_obj, "_http_client", None)
        if http_client is not None:
            http_client.timeout = auth_timeout
        admin_obj = getattr(auth_obj, "admin", None)
        if admin_obj is not None:
            admin_http_client = getattr(admin_obj, "_http_client", None)
            if admin_http_client is not None:
                admin_http_client.timeout = auth_timeout

    return client


def is_local_development() -> bool:
    """Return whether development-only endpoints may be registered."""
    return os.getenv("APP_ENV", "").strip().lower() in LOCAL_DEVELOPMENT_ENVS


def get_exception_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status from Supabase errors without exposing error data."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    return None
