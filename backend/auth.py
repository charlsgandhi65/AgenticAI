import logging
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        response = get_supabase_client(use_service_role=False).auth.get_user(
            credentials.credentials
        )
        user = response.user
    except Exception as exc:
        logger.warning("Supabase token validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Supabase access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if user is None or not user.id:
        logger.warning("Supabase token validated but returned no user or empty user id")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Supabase access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthenticatedUser(user_id=user.id)
