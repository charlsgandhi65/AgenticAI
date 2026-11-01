import asyncio
from enum import Enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class CallMode(str, Enum):
    IDLE = "IDLE"
    RINGING = "RINGING"
    AI_SCREENING = "AI_SCREENING"
    HOLD_AND_WHISPER = "HOLD_AND_WHISPER"
    AUDIO_BRIDGE = "AUDIO_BRIDGE"
    DECLINED = "DECLINED"
    ENDED = "ENDED"


@dataclass
class VoiceSession:
    session_id: str
    user_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    turn_id: int = 0
    is_ai_speaking: bool = False
    current_task: asyncio.Task[Any] | None = None
    audio_chunks: list[bytes] = field(default_factory=list)
    user_profile: dict[str, str] | None = None
    dynamic_context: str | None = None
    system_prompt: str | None = None
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_active_at = time.time()

    def add_turn(self, user_text: str, ai_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": ai_text})
        self.touch()

    def cancel_active_task(self) -> None:
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.current_task = None
        self.is_ai_speaking = False

    def clear_audio_buffer(self) -> None:
        self.audio_chunks.clear()

    def collect_audio_buffer(self) -> bytes:
        data = b"".join(self.audio_chunks)
        self.audio_chunks.clear()
        return data


class VoiceSessionManager:
    """In-memory manager for active live voice conversation sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str | None, user_id: str) -> VoiceSession:
        async with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            new_id = session_id or str(uuid.uuid4())
            session = VoiceSession(session_id=new_id, user_id=user_id)
            self._sessions[new_id] = session
            logger.info("Created new voice session %s for user %s", new_id, user_id)
            return session

    async def get_session(self, session_id: str) -> VoiceSession | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.cancel_active_task()
                logger.info("Closed voice session %s", session_id)


session_manager = VoiceSessionManager()


@dataclass
class CallChannel:
    call_id: str
    owner_user_id: str
    caller_phone_number: str
    mode: CallMode = CallMode.IDLE

    caller_ws: WebSocket | None = None
    owner_ws: WebSocket | None = None

    voice_session: VoiceSession | None = None

    caller_name: str | None = None
    caller_type: str = "UNKNOWN"
    decision: str = "HANDLE"
    decision_reason: str = ""
    is_priority: bool = False
    whisper_summary: str = ""

    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_active_at = time.time()


class CallChannelManager:
    """Manages active call channels coordinating mobile caller and laptop owner WebSockets."""

    def __init__(self) -> None:
        self._channels: dict[str, CallChannel] = {}
        self._owner_channels: dict[str, str] = {}  # owner_user_id -> call_id
        self._lock = asyncio.Lock()

    async def get_or_create_channel(
        self,
        call_id: str | None,
        owner_user_id: str,
        caller_phone_number: str,
    ) -> CallChannel:
        async with self._lock:
            cid = call_id or f"call-{uuid.uuid4().hex[:8]}"
            if cid in self._channels:
                ch = self._channels[cid]
                ch.touch()
                return ch

            channel = CallChannel(
                call_id=cid,
                owner_user_id=owner_user_id,
                caller_phone_number=caller_phone_number,
            )
            self._channels[cid] = channel
            self._owner_channels[owner_user_id] = cid
            logger.info("Created call channel %s (owner: %s, caller: %s)", cid, owner_user_id, caller_phone_number)
            return channel

    async def get_channel(self, call_id: str) -> CallChannel | None:
        async with self._lock:
            return self._channels.get(call_id)

    async def get_channel_by_owner(self, owner_user_id: str) -> CallChannel | None:
        async with self._lock:
            cid = self._owner_channels.get(owner_user_id)
            if cid:
                return self._channels.get(cid)
            return None

    async def remove_channel(self, call_id: str) -> None:
        async with self._lock:
            ch = self._channels.pop(call_id, None)
            if ch:
                if ch.owner_user_id in self._owner_channels and self._owner_channels[ch.owner_user_id] == call_id:
                    self._owner_channels.pop(ch.owner_user_id, None)
                if ch.voice_session:
                    ch.voice_session.cancel_active_task()
                logger.info("Closed call channel %s", call_id)


call_channel_manager = CallChannelManager()
