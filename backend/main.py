import logging
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import AuthenticatedUser, get_current_user
from call_decision_service import CallDecisionResponse, decide_call
from ollama_client import generate_chat_response
from prescreening_service import (
    DatabaseQueryError,
    PreScreeningContext,
    get_prescreening_context,
)
from supabase_client import get_exception_status_code, get_supabase_client, is_local_development
from user_service import (
    SupabaseRateLimitError,
    create_admin_test_user_profile,
    create_auth_user_profile,
)
from voice_service import preload_voice_models
from voice_ws import handle_voice_websocket
from call_hub import handle_call_hub_websocket

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Calling Agent Backend")

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_voice_models():
    """Warm up voice models on server startup."""
    preload_voice_models()


class ChatRequest(BaseModel):
    message: str


class SignupRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    assistant_name: str


class DevelopmentTestUserRequest(SignupRequest):
    pass


class PreScreeningRequest(BaseModel):
    caller_phone_number: str


class CallDecisionRequest(BaseModel):
    caller_phone_number: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AI Calling Agent backend is running."}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/me")
def auth_me(current_user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, str]:
    return {"user_id": current_user.user_id}


@app.post("/auth/signup", status_code=201)
def auth_signup(payload: SignupRequest) -> dict[str, object]:
    try:
        return create_auth_user_profile(
            email=payload.email,
            password=payload.password,
            first_name=payload.first_name,
            last_name=payload.last_name,
            assistant_name=payload.assistant_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, SupabaseRateLimitError) or get_exception_status_code(exc) == 429:
            raise HTTPException(
                status_code=429,
                detail="Supabase rate limit reached. Please wait and try again.",
            ) from exc
        raise HTTPException(status_code=400, detail="Supabase Auth signup failed.") from exc


if is_local_development():

    @app.post("/auth/dev/test-user", status_code=201)
    def create_development_test_user(
        payload: DevelopmentTestUserRequest,
    ) -> dict[str, object]:
        try:
            return create_admin_test_user_profile(
                email=payload.email,
                password=payload.password,
                first_name=payload.first_name,
                last_name=payload.last_name,
                assistant_name=payload.assistant_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SupabaseRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail="Supabase rate limit reached. Please wait and try again.",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            if get_exception_status_code(exc) == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Supabase rate limit reached. Please wait and try again.",
                ) from exc
            raise HTTPException(
                status_code=500,
                detail="Development test user creation failed.",
            ) from exc


@app.post("/prescreening/test", response_model=PreScreeningContext)
async def prescreening_test(
    payload: PreScreeningRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> PreScreeningContext:
    try:
        return await get_prescreening_context(
            user_id=current_user.user_id,
            caller_phone_number=payload.caller_phone_number,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatabaseQueryError as exc:
        logger.exception(
            "Database error during prescreening for user %s: %s",
            current_user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail="Database service error during pre-screening retrieval.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error during prescreening for user %s: %s",
            current_user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Pre-screening context retrieval failed.",
        ) from exc


@app.post("/ai/decide-call", response_model=CallDecisionResponse)
async def ai_decide_call(
    payload: CallDecisionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CallDecisionResponse:
    try:
        context = await get_prescreening_context(
            user_id=current_user.user_id,
            caller_phone_number=payload.caller_phone_number,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatabaseQueryError as exc:
        logger.exception(
            "Database error during call decision retrieval for user %s: %s",
            current_user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail="Database service error during pre-screening retrieval.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error during call decision pre-screening for user %s: %s",
            current_user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Pre-screening context retrieval failed.",
        ) from exc

    return await decide_call(context)


@app.post("/ai/chat")
def chat_with_qwen(payload: ChatRequest) -> dict[str, str]:
    try:
        reply = generate_chat_response(payload.message)
        return {"response": reply}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {str(exc)}") from exc


@app.get("/supabase/users-test")
def supabase_users_test() -> dict[str, object]:
    try:
        supabase = get_supabase_client(use_service_role=False)
        response = supabase.table("users").select("*").limit(1).execute()
        users = response.data or []

        return {
            "status": "ok",
            "mode": "anon",
            "message": "Supabase anon query successful.",
            "row_count": len(users),
            "sample": users[0] if users else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Supabase query failed: {str(exc)}") from exc


@app.get("/supabase/users-test-admin")
def supabase_users_test_admin() -> dict[str, object]:
    try:
        supabase = get_supabase_client(use_service_role=True)
        response = supabase.table("users").select("*").limit(1).execute()
        users = response.data or []

        return {
            "status": "ok",
            "mode": "service_role",
            "message": "Supabase service-role query successful.",
            "row_count": len(users),
            "sample": users[0] if users else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Supabase admin query failed: {str(exc)}") from exc


@app.websocket("/ws/voice-conversation")
async def websocket_voice_conversation(
    websocket: WebSocket,
    token: str | None = None,
    session_id: str | None = None,
    caller_phone_number: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    assistant_name: str | None = None,
) -> None:
    """Live browser-based two-way voice conversation endpoint."""
    await handle_voice_websocket(
        websocket,
        token=token,
        session_id=session_id,
        caller_phone_number=caller_phone_number,
        first_name=first_name,
        last_name=last_name,
        assistant_name=assistant_name,
    )


@app.websocket("/ws/call-hub")
async def websocket_call_hub(
    websocket: WebSocket,
    role: str = "caller",
    token: str | None = None,
    call_id: str | None = None,
    caller_phone_number: str | None = None,
    owner_user_id: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    assistant_name: str | None = None,
) -> None:
    """Two-device calling simulation hub endpoint for mobile caller and laptop owner."""
    await handle_call_hub_websocket(
        websocket=websocket,
        role=role,
        token=token,
        call_id=call_id,
        caller_phone_number=caller_phone_number,
        owner_user_id=owner_user_id,
        first_name=first_name,
        last_name=last_name,
        assistant_name=assistant_name,
    )


_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/voice", StaticFiles(directory=str(_frontend_dist), html=True), name="voice")
