import asyncio
import json
import logging
import struct
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from call_decision_service import decide_call
from prescreening_service import get_prescreening_context
from supabase_client import get_supabase_client, is_local_development
from system_prompt import build_system_prompt, fetch_dynamic_context, fetch_user_profile, is_valid_uuid
from voice_service import (
    generate_voice_response,
    synthesize_text_to_wav,
    transcribe_audio_bytes,
)
from voice_session_manager import (
    CallChannel,
    CallMode,
    VoiceSession,
    call_channel_manager,
    session_manager,
)
from voice_ws import authenticate_ws

logger = logging.getLogger(__name__)


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wraps raw 16-bit linear PCM audio in a standard 44-byte WAV container."""
    header = bytearray(44)
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, 36 + len(pcm_bytes))
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)
    struct.pack_into("<H", header, 20, 1)  # PCM
    struct.pack_into("<H", header, 22, 1)  # Mono
    struct.pack_into("<I", header, 24, sample_rate)
    struct.pack_into("<I", header, 28, sample_rate * 2)
    struct.pack_into("<H", header, 32, 2)
    struct.pack_into("<H", header, 34, 16)
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, len(pcm_bytes))
    return bytes(header) + pcm_bytes


async def safe_send_json(ws: WebSocket | None, data: dict[str, Any]) -> None:
    """Safely send a JSON message to a WebSocket if connected."""
    if ws:
        try:
            await ws.send_json(data)
        except Exception as exc:
            logger.debug("Failed to send JSON to WebSocket: %s", exc)


async def safe_send_bytes(ws: WebSocket | None, data: bytes) -> None:
    """Safely send binary bytes to a WebSocket if connected."""
    if ws and data:
        try:
            await ws.send_bytes(data)
        except Exception as exc:
            logger.debug("Failed to send bytes to WebSocket: %s", exc)


async def execute_caller_ai_turn(
    channel: CallChannel,
    audio_bytes: bytes,
    turn_id: int,
) -> None:
    """Processes a caller speech turn: STT -> Qwen -> Piper -> Send audio to caller & broadcast transcript to owner."""
    session = channel.voice_session
    if not session:
        return

    async with session.turn_lock:
        turn_start_time = time.perf_counter()
        buffer_duration = len(audio_bytes) / 32000.0
        logger.info(
            "[Call %s | Turn %d] Caller speech processing started (%d bytes, ~%.2fs audio)",
            channel.call_id,
            turn_id,
            len(audio_bytes),
            buffer_duration,
        )

        try:
            # 1. Notify state
            await safe_send_json(channel.caller_ws, {"type": "state", "state": "processing", "turn_id": turn_id})
            await safe_send_json(channel.owner_ws, {"type": "caller_state", "state": "processing", "turn_id": turn_id})

            # 2. STT via faster-whisper
            t0 = time.perf_counter()
            user_transcript = await asyncio.to_thread(transcribe_audio_bytes, audio_bytes)
            stt_elapsed = time.perf_counter() - t0
            logger.info(
                "[Call %s | Turn %d] STT transcription completed: %r (elapsed: %.2fs)",
                channel.call_id,
                turn_id,
                user_transcript,
                stt_elapsed,
            )

            if not user_transcript or not user_transcript.strip():
                logger.info("[Call %s | Turn %d] Empty transcription. Returning to listening.", channel.call_id, turn_id)
                await safe_send_json(channel.caller_ws, {"type": "state", "state": "listening", "turn_id": turn_id})
                return

            # Broadcast Caller Transcript to BOTH Caller and Owner
            transcript_msg = {
                "type": "transcript",
                "role": "user",
                "text": user_transcript,
                "turn_id": turn_id,
                "caller_number": channel.caller_phone_number,
                "caller_name": channel.caller_name or "Caller",
            }
            await safe_send_json(channel.caller_ws, transcript_msg)
            await safe_send_json(channel.owner_ws, transcript_msg)

            # Check for Escalation Intent from user transcript
            transcript_lower = user_transcript.lower()
            needs_escalation = (
                "speak to" in transcript_lower
                or "talk to" in transcript_lower
                or "speak with" in transcript_lower
                or "connect me" in transcript_lower
                or "transfer me" in transcript_lower
                or "financial transaction" in transcript_lower
                or "urgent" in transcript_lower
                or "emergency" in transcript_lower
            )

            owner_first_name = (session.user_profile.get("first_name") or "the owner").strip()

            # 3. LLM Response via Qwen
            t1 = time.perf_counter()
            ai_reply = await asyncio.to_thread(
                generate_voice_response,
                list(session.history),
                user_transcript,
                session.system_prompt,
            )
            llm_elapsed = time.perf_counter() - t1
            logger.info(
                "[Call %s | Turn %d] Qwen response: %r (elapsed: %.2fs)",
                channel.call_id,
                turn_id,
                ai_reply,
                llm_elapsed,
            )

            if not ai_reply or not ai_reply.strip():
                ai_reply = "I understand. How can I assist you with that?"

            # If model response or user input indicates escalation
            if needs_escalation or "hold for a moment" in ai_reply.lower() or "connect you" in ai_reply.lower():
                channel.mode = CallMode.HOLD_AND_WHISPER
                logger.info("[Call %s] Escalation detected. Switching to HOLD_AND_WHISPER", channel.call_id)

                # Commit turn to history
                session.add_turn(user_transcript, ai_reply)

                # Send AI transcript to caller and owner
                ai_transcript_msg = {
                    "type": "transcript",
                    "role": "assistant",
                    "text": ai_reply,
                    "turn_id": turn_id,
                }
                await safe_send_json(channel.caller_ws, ai_transcript_msg)
                await safe_send_json(channel.owner_ws, ai_transcript_msg)

                # Deliver hold audio to caller
                wav_bytes = await asyncio.to_thread(synthesize_text_to_wav, ai_reply)
                if wav_bytes:
                    session.is_ai_speaking = True
                    await safe_send_json(channel.caller_ws, {
                        "type": "ai_audio_start",
                        "turn_id": turn_id,
                        "format": "wav",
                        "byte_length": len(wav_bytes),
                    })
                    await safe_send_bytes(channel.caller_ws, wav_bytes)
                    await safe_send_json(channel.caller_ws, {"type": "state", "state": "speaking", "turn_id": turn_id})

                # Broadcast Private Whisper Alert to Owner
                whisper_text = (
                    f"Private Whisper: Caller {channel.caller_name or channel.caller_phone_number} "
                    f"is requesting to discuss: '{user_transcript}'. "
                    f"Tap Accept to take over the call or Reject to have AI take a message."
                )
                channel.whisper_summary = whisper_text
                await safe_send_json(channel.owner_ws, {
                    "type": "whisper_escalation",
                    "call_id": channel.call_id,
                    "caller_number": channel.caller_phone_number,
                    "caller_name": channel.caller_name,
                    "whisper_text": whisper_text,
                    "requires_takeover": True,
                })
                await safe_send_json(channel.caller_ws, {
                    "type": "call_mode",
                    "mode": "HOLD_AND_WHISPER",
                    "message": f"Please hold while I check if {owner_first_name} is available...",
                })
                return

            # Normal AI Conversation Handling
            session.add_turn(user_transcript, ai_reply)

            # Send Assistant Transcript to BOTH Caller and Owner
            ai_transcript_msg = {
                "type": "transcript",
                "role": "assistant",
                "text": ai_reply,
                "turn_id": turn_id,
            }
            await safe_send_json(channel.caller_ws, ai_transcript_msg)
            await safe_send_json(channel.owner_ws, ai_transcript_msg)

            # Synthesize Piper TTS and send WAV to Caller ONLY
            t2 = time.perf_counter()
            wav_bytes = await asyncio.to_thread(synthesize_text_to_wav, ai_reply)
            tts_elapsed = time.perf_counter() - t2

            if wav_bytes:
                session.is_ai_speaking = True
                await safe_send_json(channel.caller_ws, {
                    "type": "ai_audio_start",
                    "turn_id": turn_id,
                    "format": "wav",
                    "byte_length": len(wav_bytes),
                })
                await safe_send_bytes(channel.caller_ws, wav_bytes)
                await safe_send_json(channel.caller_ws, {"type": "state", "state": "speaking", "turn_id": turn_id})
            else:
                await safe_send_json(channel.caller_ws, {"type": "state", "state": "listening", "turn_id": turn_id})

        except Exception as exc:
            logger.exception("[Call %s | Turn %d] Error in AI turn: %s", channel.call_id, turn_id, exc)


async def handle_call_hub_websocket(
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
    """Unified Call Hub WebSocket supporting mobile Caller and laptop Owner roles with audio bridging."""
    await websocket.accept()

    # Normalize role
    role_clean = role.lower().strip()
    if role_clean not in ("caller", "owner"):
        role_clean = "caller"

    # Authenticate user if owner or token provided
    user_id = "dev-local-user"
    if token:
        try:
            user_id = await authenticate_ws(websocket, token)
        except Exception:
            user_id = "dev-local-user"

    # -------------------------------------------------------------
    # ROLE: OWNER (Laptop Interface)
    # -------------------------------------------------------------
    if role_clean == "owner":
        owner_id = (owner_user_id or user_id or "dev-local-user").strip()
        logger.info("[CallHub] Owner connected: user_id=%s", owner_id)

        # Load profile
        profile = await asyncio.to_thread(fetch_user_profile, owner_id) if is_valid_uuid(owner_id) else None
        if not profile:
            profile = {
                "first_name": (first_name or "Charles").strip(),
                "last_name": (last_name or "Smith").strip(),
                "assistant_name": (assistant_name or "Alaana").strip(),
            }

        # Check existing channel or register owner
        active_channel = await call_channel_manager.get_channel_by_owner(owner_id)
        if not active_channel:
            active_channel = await call_channel_manager.get_or_create_channel(
                call_id=call_id,
                owner_user_id=owner_id,
                caller_phone_number="Waiting for call...",
            )
        active_channel.owner_ws = websocket
        active_channel.touch()

        # Send owner ready event
        await safe_send_json(websocket, {
            "type": "owner_ready",
            "owner_user_id": owner_id,
            "call_id": active_channel.call_id,
            "first_name": profile.get("first_name", ""),
            "last_name": profile.get("last_name", ""),
            "assistant_name": profile.get("assistant_name", "Alaana"),
            "call_mode": active_channel.mode.value,
            "caller_number": active_channel.caller_phone_number,
            "caller_name": active_channel.caller_name,
            "caller_type": active_channel.caller_type,
            "decision": active_channel.decision,
            "decision_reason": active_channel.decision_reason,
            "is_priority": active_channel.is_priority,
        })

        try:
            while True:
                message = await websocket.receive()
                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    break

                # Binary audio from Owner (when in AUDIO_BRIDGE mode)
                if "bytes" in message and message["bytes"]:
                    chunk = message["bytes"]
                    if active_channel.mode == CallMode.AUDIO_BRIDGE and active_channel.caller_ws:
                        # Forward audio chunk straight to mobile caller phone speaker!
                        wav_frame = pcm16_to_wav(chunk, sample_rate=16000)
                        await safe_send_bytes(active_channel.caller_ws, wav_frame)
                    continue

                if "text" in message and message["text"]:
                    try:
                        payload = json.loads(message["text"])
                    except Exception:
                        continue

                    event = payload.get("type", "")

                    # 1. Owner accepts call transfer / takes over call
                    if event in ("accept_transfer", "takeover_call"):
                        active_channel.mode = CallMode.AUDIO_BRIDGE
                        active_channel.is_bridge_active = True
                        if active_channel.voice_session:
                            active_channel.voice_session.cancel_active_task()

                        logger.info("[Call %s] Owner ACCEPTED transfer. Audio bridge ACTIVATED.", active_channel.call_id)

                        # Notify Owner
                        await safe_send_json(websocket, {
                            "type": "bridge_connected",
                            "call_id": active_channel.call_id,
                            "mode": "AUDIO_BRIDGE",
                            "message": "Connected! You are now speaking directly with the caller.",
                        })

                        # Notify Caller
                        owner_fn = profile.get("first_name", "the user")
                        await safe_send_json(active_channel.caller_ws, {
                            "type": "call_transferred",
                            "call_id": active_channel.call_id,
                            "mode": "AUDIO_BRIDGE",
                            "message": f"Connected with {owner_fn}. Speak naturally.",
                        })

                    # 2. Owner declines / rejects transfer
                    elif event in ("reject_transfer", "decline_transfer"):
                        active_channel.mode = CallMode.AI_SCREENING
                        active_channel.is_bridge_active = False
                        logger.info("[Call %s] Owner REJECTED transfer. AI fallback message taking.", active_channel.call_id)

                        owner_fn = profile.get("first_name", "the user")
                        fallback_text = (
                            f"I apologize, {owner_fn} is currently unavailable to take your call. "
                            f"Would you like to leave a message for him?"
                        )

                        # Notify Owner
                        await safe_send_json(websocket, {
                            "type": "transfer_rejected",
                            "call_id": active_channel.call_id,
                            "mode": "AI_SCREENING",
                            "message": f"Transfer rejected. AI is offering to take a message.",
                        })

                        # Notify Caller and speak fallback message
                        if active_channel.caller_ws:
                            await safe_send_json(active_channel.caller_ws, {
                                "type": "transcript",
                                "role": "assistant",
                                "text": fallback_text,
                                "turn_id": active_channel.voice_session.turn_id if active_channel.voice_session else 1,
                            })
                            fallback_wav = await asyncio.to_thread(synthesize_text_to_wav, fallback_text)
                            if fallback_wav:
                                await safe_send_json(active_channel.caller_ws, {
                                    "type": "ai_audio_start",
                                    "turn_id": 999,
                                    "format": "wav",
                                    "byte_length": len(fallback_wav),
                                })
                                await safe_send_bytes(active_channel.caller_ws, fallback_wav)
                                await safe_send_json(active_channel.caller_ws, {"type": "state", "state": "speaking"})

                    # 3. Owner ends call
                    elif event == "end_call":
                        active_channel.mode = CallMode.ENDED
                        active_channel.is_bridge_active = False
                        logger.info("[Call %s] Owner terminated call.", active_channel.call_id)

                        await safe_send_json(active_channel.caller_ws, {
                            "type": "call_ended",
                            "call_id": active_channel.call_id,
                            "reason": "Owner ended the call.",
                        })
                        await safe_send_json(websocket, {
                            "type": "call_ended",
                            "call_id": active_channel.call_id,
                            "reason": "Call ended.",
                        })
                        await call_channel_manager.remove_channel(active_channel.call_id)

        except WebSocketDisconnect:
            logger.info("Owner WebSocket disconnected for user %s", owner_id)
        except Exception as exc:
            logger.exception("Owner WebSocket error: %s", exc)
        finally:
            if active_channel and active_channel.owner_ws == websocket:
                active_channel.owner_ws = None
        return

    # -------------------------------------------------------------
    # ROLE: CALLER (Mobile Phone Interface)
    # -------------------------------------------------------------
    caller_num = (caller_phone_number or "9824378001").strip()
    target_owner = (owner_user_id or user_id or "dev-local-user").strip()

    logger.info("[CallHub] Incoming caller connection: phone=%s, target_owner=%s", caller_num, target_owner)

    # Resolve target owner's active channel if present
    owner_channel = await call_channel_manager.get_channel_by_owner(target_owner)
    assigned_call_id = call_id or (owner_channel.call_id if owner_channel else None)

    channel = await call_channel_manager.get_or_create_channel(
        call_id=assigned_call_id,
        owner_user_id=target_owner,
        caller_phone_number=caller_num,
    )
    channel.caller_ws = websocket
    channel.caller_phone_number = caller_num
    channel.touch()

    # Load owner profile for assistant personalization
    owner_profile = await asyncio.to_thread(fetch_user_profile, channel.owner_user_id) if is_valid_uuid(channel.owner_user_id) else None
    if not owner_profile:
        owner_profile = {
            "first_name": (first_name or "Charles").strip(),
            "last_name": (last_name or "Smith").strip(),
            "assistant_name": (assistant_name or "Alaana").strip(),
        }

    owner_fn = owner_profile.get("first_name", "Charles")
    owner_ln = owner_profile.get("last_name", "Smith")
    assistant_fn = owner_profile.get("assistant_name", "Alaana")

    # Initialize AI voice session for this channel
    voice_session = await session_manager.get_or_create(f"voice-{channel.call_id}", channel.owner_user_id)
    voice_session.user_profile = owner_profile
    dynamic_ctx = await asyncio.to_thread(fetch_dynamic_context, channel.owner_user_id, caller_num)
    voice_session.dynamic_context = dynamic_ctx
    voice_session.system_prompt = build_system_prompt(owner_profile, dynamic_ctx)
    channel.voice_session = voice_session

    # -------------------------------------------------------------
    # MANDATORY CALLER ROUTING (Requirement 4)
    # 1. Check Priority Contacts first
    # 2. If Priority Contact -> Direct connection to owner without AI
    # 3. Otherwise -> Run Decision Engine (HANDLE / DECLINE / ESCALATE)
    # -------------------------------------------------------------
    try:
        prescreen_ctx = await get_prescreening_context(channel.owner_user_id, caller_num)
        is_prio = bool(prescreen_ctx.priority_contacts)
    except Exception as exc:
        logger.warning("[Call %s] Prescreening lookup error: %s", channel.call_id, exc)
        prescreen_ctx = None
        is_prio = False

    # 1. PRIORITY CONTACT ROUTING (Direct Owner Connection without AI)
    if is_prio:
        contact = prescreen_ctx.priority_contacts[0]
        c_name = contact.get("contact_name") if isinstance(contact, dict) else getattr(contact, "contact_name", "Priority Contact")
        c_cat = contact.get("category") if isinstance(contact, dict) else getattr(contact, "category", "GENERAL")
        channel.is_priority = True
        channel.caller_name = c_name or "Priority Contact"
        channel.caller_type = "PRIORITY_CONTACT"
        channel.decision = "ESCALATE"
        channel.decision_reason = f"Priority contact ({channel.caller_name}, Category: {c_cat}). Directly connecting to owner without AI intervention."
        channel.mode = CallMode.HOLD_AND_WHISPER
        channel.is_bridge_active = False

        logger.info("[Call %s] Priority Contact detected: %s. Direct connection.", channel.call_id, channel.caller_name)

        # Notify Caller
        await safe_send_json(websocket, {
            "type": "call_ringing",
            "call_id": channel.call_id,
            "mode": "HOLD_AND_WHISPER",
            "caller_name": channel.caller_name,
            "is_priority": True,
            "message": f"Priority caller recognized. Connecting you directly to {owner_fn}...",
        })

        # Synthesize direct connection audio announcement to caller
        direct_announcement = f"Connecting you directly to {owner_fn}. Please stay on the line."
        ann_wav = await asyncio.to_thread(synthesize_text_to_wav, direct_announcement)
        if ann_wav:
            await safe_send_json(websocket, {
                "type": "ai_audio_start",
                "turn_id": 0,
                "format": "wav",
                "byte_length": len(ann_wav),
            })
            await safe_send_bytes(websocket, ann_wav)

        # Notify Owner with Urgent Priority Ringing Alert
        await safe_send_json(channel.owner_ws, {
            "type": "incoming_call",
            "call_id": channel.call_id,
            "caller_number": caller_num,
            "caller_name": channel.caller_name,
            "caller_type": "PRIORITY_CONTACT",
            "category": c_cat,
            "decision": "ESCALATE",
            "decision_reason": channel.decision_reason,
            "is_priority": True,
            "requires_takeover": True,
            "message": f"PRIORITY CALL from {channel.caller_name} ({c_cat}). Tap Accept to speak directly.",
        })

    # 2. NON-PRIORITY: Execute Decision Engine
    else:
        try:
            if prescreen_ctx:
                dec_resp = await decide_call(prescreen_ctx)
                channel.caller_type = dec_resp.caller_type.value
                channel.decision = dec_resp.decision.value
                channel.decision_reason = dec_resp.reason
                if prescreen_ctx.contacts:
                    c0 = prescreen_ctx.contacts[0]
                    channel.caller_name = c0.get("contact_name") if isinstance(c0, dict) else getattr(c0, "contact_name", "Unknown Caller")
                else:
                    channel.caller_name = "Unknown Caller"
            else:
                channel.caller_type = "UNKNOWN"
                channel.decision = "HANDLE"
                channel.decision_reason = "Default to AI handling for caller evaluation."
                channel.caller_name = "Unknown Caller"
        except Exception as exc:
            logger.warning("[Call %s] Decision engine fallback to HANDLE: %s", channel.call_id, exc)
            channel.caller_type = "UNKNOWN"
            channel.decision = "HANDLE"
            channel.decision_reason = "Fallback to AI handling."

        logger.info(
            "[Call %s] Caller classification: %s | Decision: %s (%s)",
            channel.call_id,
            channel.caller_type,
            channel.decision,
            channel.decision_reason,
        )

        # Broadcast incoming call event to Owner Dashboard
        await safe_send_json(channel.owner_ws, {
            "type": "incoming_call",
            "call_id": channel.call_id,
            "caller_number": caller_num,
            "caller_name": channel.caller_name,
            "caller_type": channel.caller_type,
            "decision": channel.decision,
            "decision_reason": channel.decision_reason,
            "is_priority": False,
            "ai_handling": channel.decision == "HANDLE",
            "requires_takeover": channel.decision == "ESCALATE",
        })

        # A. DECLINE -> Reject call
        if channel.decision == "DECLINE":
            channel.mode = CallMode.DECLINED
            decline_msg = "The user is not accepting calls from this number at this time. Goodbye."
            await safe_send_json(websocket, {
                "type": "call_declined",
                "call_id": channel.call_id,
                "message": decline_msg,
            })
            dec_wav = await asyncio.to_thread(synthesize_text_to_wav, decline_msg)
            if dec_wav:
                await safe_send_json(websocket, {
                    "type": "ai_audio_start",
                    "turn_id": 0,
                    "format": "wav",
                    "byte_length": len(dec_wav),
                })
                await safe_send_bytes(websocket, dec_wav)

        # B. ESCALATE -> Put caller on hold immediately and prompt owner
        elif channel.decision == "ESCALATE":
            channel.mode = CallMode.HOLD_AND_WHISPER
            hold_msg = f"Please hold for a moment while I try to connect you with {owner_fn}."
            await safe_send_json(websocket, {
                "type": "call_mode",
                "mode": "HOLD_AND_WHISPER",
                "message": hold_msg,
            })
            hold_wav = await asyncio.to_thread(synthesize_text_to_wav, hold_msg)
            if hold_wav:
                await safe_send_json(websocket, {
                    "type": "ai_audio_start",
                    "turn_id": 0,
                    "format": "wav",
                    "byte_length": len(hold_wav),
                })
                await safe_send_bytes(websocket, hold_wav)

        # C. HANDLE -> AI Receptionist takes the call
        else:
            channel.mode = CallMode.AI_SCREENING
            greeting = f"Hello! This is {assistant_fn}, {owner_fn}'s personal assistant. How can I help you today?"
            await safe_send_json(websocket, {
                "type": "call_connected",
                "call_id": channel.call_id,
                "mode": "AI_SCREENING",
                "assistant_name": assistant_fn,
                "owner_name": f"{owner_fn} {owner_ln}".strip(),
                "message": greeting,
            })

            # Broadcast greeting transcript to BOTH
            greet_transcript = {
                "type": "transcript",
                "role": "assistant",
                "text": greeting,
                "turn_id": 0,
            }
            await safe_send_json(websocket, greet_transcript)
            await safe_send_json(channel.owner_ws, greet_transcript)

            # Synthesize and send initial greeting audio to Caller phone
            greet_wav = await asyncio.to_thread(synthesize_text_to_wav, greeting)
            if greet_wav:
                channel.voice_session.is_ai_speaking = True
                await safe_send_json(websocket, {
                    "type": "ai_audio_start",
                    "turn_id": 0,
                    "format": "wav",
                    "byte_length": len(greet_wav),
                })
                await safe_send_bytes(websocket, greet_wav)
                await safe_send_json(websocket, {"type": "state", "state": "speaking", "turn_id": 0})

    # -------------------------------------------------------------
    # CALLER INBOUND MESSAGE LOOP
    # -------------------------------------------------------------
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break

            # Handle Binary PCM Audio from Caller Phone
            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]

                # If in AUDIO_BRIDGE mode -> Forward straight to Owner Laptop speakers!
                if channel.mode == CallMode.AUDIO_BRIDGE:
                    if channel.owner_ws:
                        wav_frame = pcm16_to_wav(chunk, sample_rate=16000)
                        await safe_send_bytes(channel.owner_ws, wav_frame)
                    continue

                # If in AI_SCREENING mode -> Buffer for faster-whisper STT
                if channel.mode == CallMode.AI_SCREENING and channel.voice_session:
                    channel.voice_session.audio_chunks.append(chunk)
                continue

            # Handle JSON Control Messages
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue

                event = payload.get("type", "")

                # 1. Caller finished speaking in AI mode -> Trigger STT & Qwen
                if event == "speech_end":
                    if channel.mode == CallMode.AI_SCREENING and channel.voice_session:
                        audio_data = channel.voice_session.collect_audio_buffer()
                        if not audio_data or len(audio_data) < 3200:
                            continue

                        channel.voice_session.turn_id += 1
                        t_id = channel.voice_session.turn_id
                        task = asyncio.create_task(execute_caller_ai_turn(channel, audio_data, t_id))
                        channel.voice_session.current_task = task

                # 2. Caller Barge-in / Interrupt while AI speaks
                elif event in ("interrupt", "user_speaking"):
                    if channel.mode == CallMode.AI_SCREENING and channel.voice_session and channel.voice_session.is_ai_speaking:
                        logger.info("[Call %s] Caller barge-in. Halting AI speech.", channel.call_id)
                        channel.voice_session.is_ai_speaking = False
                        channel.voice_session.cancel_active_task()
                        await safe_send_json(websocket, {"type": "interrupted", "turn_id": channel.voice_session.turn_id})
                        await safe_send_json(websocket, {"type": "state", "state": "listening"})

                # 3. Audio playback ended on phone
                elif event == "audio_playback_ended":
                    if channel.voice_session:
                        channel.voice_session.is_ai_speaking = False
                    await safe_send_json(websocket, {"type": "state", "state": "listening"})

                # 4. Caller hangs up / ends call
                elif event == "end_call":
                    channel.mode = CallMode.ENDED
                    logger.info("[Call %s] Caller hung up.", channel.call_id)
                    await safe_send_json(channel.owner_ws, {
                        "type": "call_ended",
                        "call_id": channel.call_id,
                        "reason": "Caller hung up.",
                    })
                    await safe_send_json(websocket, {
                        "type": "call_ended",
                        "call_id": channel.call_id,
                        "reason": "Call disconnected.",
                    })
                    await call_channel_manager.remove_channel(channel.call_id)
                    break

    except WebSocketDisconnect:
        logger.info("[Call %s] Caller disconnected from call hub.", channel.call_id)
    except Exception as exc:
        logger.exception("[Call %s] Caller WebSocket error: %s", channel.call_id, exc)
    finally:
        channel.mode = CallMode.ENDED
        await safe_send_json(channel.owner_ws, {
            "type": "call_ended",
            "call_id": channel.call_id,
            "reason": "Caller disconnected.",
        })
        await call_channel_manager.remove_channel(channel.call_id)
