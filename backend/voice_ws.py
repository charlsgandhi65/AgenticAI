import asyncio
import json
import logging
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from supabase_client import get_supabase_client, is_local_development
from system_prompt import (
    build_system_prompt,
    fetch_dynamic_context,
    fetch_user_profile,
    is_valid_uuid,
)
from voice_service import (
    generate_voice_response,
    synthesize_text_to_wav,
    transcribe_audio_bytes,
)
from voice_session_manager import VoiceSession, session_manager

logger = logging.getLogger(__name__)


async def authenticate_ws(websocket: WebSocket, token: str | None) -> str:
    """Validate Supabase auth token or assign development user."""
    if token:
        try:
            client = get_supabase_client(use_service_role=False)
            response = client.auth.get_user(token)
            if response and response.user and response.user.id:
                return response.user.id
        except Exception as exc:
            logger.warning("WebSocket token verification failed: %s", exc)

    # In local development or fallback, allow testing
    if is_local_development() or not token:
        logger.info("Allowing local development session without strict JWT.")
        return "dev-local-user"

    raise ValueError("Authentication required: Invalid or missing Supabase token.")


async def execute_turn(
    session: VoiceSession,
    websocket: WebSocket,
    audio_bytes: bytes,
    turn_id: int,
) -> None:
    """Processes a single conversational turn: STT -> LLM -> TTS -> Audio delivery."""
    async with session.turn_lock:
        turn_start_time = time.perf_counter()
        buffer_duration = len(audio_bytes) / 32000.0  # 16kHz 16-bit mono = 32000 bytes/sec
        logger.info(
            "[Session %s | Turn %d] Starting turn execution (audio buffer: %d bytes, ~%.2fs)",
            session.session_id,
            turn_id,
            len(audio_bytes),
            buffer_duration,
        )

        try:
            # 1. State: Processing
            await websocket.send_json({
                "type": "state",
                "state": "processing",
                "turn_id": turn_id,
            })

            # 2. STT via faster-whisper
            t0 = time.perf_counter()
            logger.info("[Session %s | Turn %d] STT transcription started with faster-whisper", session.session_id, turn_id)
            user_transcript = await asyncio.to_thread(transcribe_audio_bytes, audio_bytes)
            stt_elapsed = time.perf_counter() - t0
            logger.info(
                "[Session %s | Turn %d] STT transcription completed: %r (elapsed: %.2fs)",
                session.session_id,
                turn_id,
                user_transcript,
                stt_elapsed,
            )

            if not user_transcript or not user_transcript.strip():
                logger.info("[Session %s | Turn %d] STT produced empty text. Returning to listening.", session.session_id, turn_id)
                await websocket.send_json({
                    "type": "state",
                    "state": "listening",
                    "turn_id": turn_id,
                })
                return

            # Send recognized user transcript to frontend
            await websocket.send_json({
                "type": "transcript",
                "role": "user",
                "text": user_transcript,
                "turn_id": turn_id,
            })

            # 3. LLM response via Qwen / Ollama
            logger.info("[Session %s | Turn %d] User message submitted to Qwen: %r", session.session_id, turn_id, user_transcript)
            history_summary = [{"role": m["role"], "content": m["content"][:40]} for m in session.history]
            logger.info(
                "[Session %s | Turn %d] Conversation history sent to Qwen (%d turns): %s",
                session.session_id,
                turn_id,
                len(session.history) // 2,
                history_summary,
            )

            t1 = time.perf_counter()
            ai_reply = await asyncio.to_thread(
                generate_voice_response,
                list(session.history),
                user_transcript,
                session.system_prompt,
            )
            llm_elapsed = time.perf_counter() - t1
            logger.info(
                "[Session %s | Turn %d] Qwen response received: %r (elapsed: %.2fs)",
                session.session_id,
                turn_id,
                ai_reply,
                llm_elapsed,
            )

            if not ai_reply or not ai_reply.strip():
                ai_reply = "I understand. How else can I help you today?"
                logger.warning("[Session %s | Turn %d] Empty Qwen response; applied fallback: %r", session.session_id, turn_id, ai_reply)

            # Commit immediately to multi-turn conversation memory
            session.add_turn(user_transcript, ai_reply)
            logger.info(
                "[Session %s | Turn %d] Committed turn to conversation history. History now contains %d turns.",
                session.session_id,
                turn_id,
                len(session.history) // 2,
            )

            # Send AI response text to frontend
            await websocket.send_json({
                "type": "transcript",
                "role": "assistant",
                "text": ai_reply,
                "turn_id": turn_id,
            })

            # 4. TTS synthesis via Piper
            t2 = time.perf_counter()
            logger.info("[Session %s | Turn %d] Piper synthesis started for text length %d chars", session.session_id, turn_id, len(ai_reply))
            wav_bytes = await asyncio.to_thread(synthesize_text_to_wav, ai_reply)
            tts_elapsed = time.perf_counter() - t2
            logger.info(
                "[Session %s | Turn %d] Piper synthesis completed: %d WAV bytes (elapsed: %.2fs)",
                session.session_id,
                turn_id,
                len(wav_bytes) if wav_bytes else 0,
                tts_elapsed,
            )

            if not wav_bytes:
                logger.warning("[Session %s | Turn %d] TTS produced empty audio.", session.session_id, turn_id)
                await websocket.send_json({
                    "type": "state",
                    "state": "listening",
                    "turn_id": turn_id,
                })
                return

            # 5. Deliver binary audio to frontend
            total_elapsed = time.perf_counter() - turn_start_time
            session.is_ai_speaking = True
            logger.info(
                "[Session %s | Turn %d] Audio response delivered to frontend: %d bytes (total turn elapsed: %.2fs)",
                session.session_id,
                turn_id,
                len(wav_bytes),
                total_elapsed,
            )

            await websocket.send_json({
                "type": "ai_audio_start",
                "turn_id": turn_id,
                "format": "wav",
                "byte_length": len(wav_bytes),
            })

            # Send binary WAV audio frame
            await websocket.send_bytes(wav_bytes)

            await websocket.send_json({
                "type": "state",
                "state": "speaking",
                "turn_id": turn_id,
            })

        except asyncio.CancelledError:
            logger.info("[Session %s | Turn %d] Turn task was cancelled.", session.session_id, turn_id)
        except Exception as exc:
            logger.exception("[Session %s | Turn %d] Error during voice turn: %s", session.session_id, turn_id, exc)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(exc)}",
                    "turn_id": turn_id,
                })
                await websocket.send_json({
                    "type": "state",
                    "state": "listening",
                    "turn_id": turn_id,
                })
            except Exception:
                pass


async def handle_voice_websocket(
    websocket: WebSocket,
    token: str | None = None,
    session_id: str | None = None,
    caller_phone_number: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    assistant_name: str | None = None,
) -> None:
    """WebSocket endpoint handler for live two-way browser voice conversation."""
    await websocket.accept()

    try:
        user_id = await authenticate_ws(websocket, token)
    except Exception as exc:
        logger.warning("Rejecting unauthenticated WebSocket: %s", exc)
        await websocket.send_json({"type": "error", "message": "Authentication failed"})
        await websocket.close(code=4401)
        return

    session = await session_manager.get_or_create(session_id, user_id)

    # Initialize user profile & system prompt
    db_profile = await asyncio.to_thread(fetch_user_profile, session.user_id) if is_valid_uuid(session.user_id) else None

    # Priority: DB profile if exists; fall back to query params if provided
    profile: dict[str, str] = {}
    if db_profile:
        profile = dict(db_profile)

    # Allow query parameter overrides if supplied non-empty
    if first_name and first_name.strip():
        profile["first_name"] = first_name.strip()
    if last_name and last_name.strip():
        profile["last_name"] = last_name.strip()
    if assistant_name and assistant_name.strip():
        profile["assistant_name"] = assistant_name.strip()

    session.user_profile = profile
    dynamic_context = await asyncio.to_thread(
        fetch_dynamic_context, session.user_id, caller_phone_number
    )
    session.dynamic_context = dynamic_context
    session.system_prompt = build_system_prompt(session.user_profile, dynamic_context)

    logger.info(
        "[Session %s] Calling Agent initialized: User='%s %s', Assistant='%s' (user_id=%s)",
        session.session_id,
        session.user_profile.get("first_name", ""),
        session.user_profile.get("last_name", ""),
        session.user_profile.get("assistant_name", ""),
        session.user_id,
    )

    # Notify frontend that session is ready with full identity
    await websocket.send_json({
        "type": "session_ready",
        "session_id": session.session_id,
        "user_id": session.user_id,
        "first_name": session.user_profile.get("first_name", ""),
        "last_name": session.user_profile.get("last_name", ""),
        "assistant_name": session.user_profile.get("assistant_name", "") or "Assistant",
        "history": session.history,
        "state": "listening",
    })

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            # Handle Binary Audio Chunks
            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]
                if not session.audio_chunks:
                    logger.info("[Session %s | Turn %d] Speech started: receiving audio stream", session.session_id, session.turn_id + 1)
                session.audio_chunks.append(chunk)
                continue

            # Handle JSON Control & Text Messages
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue

                event_type = payload.get("type", "")

                # 1. Barge-in / Interrupt: User spoke while AI was speaking OR manual interrupt
                if event_type == "interrupt" or (event_type == "user_speaking" and session.is_ai_speaking):
                    logger.info(
                        "[Session %s | Turn %d] Barge-in/Interrupt triggered (event=%s, is_ai_speaking=%s). Stopping AI audio.",
                        session.session_id,
                        session.turn_id,
                        event_type,
                        session.is_ai_speaking,
                    )
                    session.is_ai_speaking = False
                    session.turn_id += 1
                    session.cancel_active_task()
                    await websocket.send_json({
                        "type": "interrupted",
                        "turn_id": session.turn_id,
                    })
                    await websocket.send_json({
                        "type": "state",
                        "state": "listening",
                        "turn_id": session.turn_id,
                    })

                # 2. User finished speaking (silence detected by VAD)
                elif event_type == "speech_end":
                    audio_data = session.collect_audio_buffer()
                    if not audio_data or len(audio_data) < 3200:  # less than 100ms
                        logger.info(
                            "[Session %s] Ignoring tiny speech_end buffer (%d bytes < 100ms)",
                            session.session_id,
                            len(audio_data) if audio_data else 0,
                        )
                        continue

                    duration_sec = len(audio_data) / 32000.0
                    session.turn_id += 1
                    current_turn = session.turn_id
                    logger.info(
                        "[Session %s | Turn %d] Speech ended: collected audio buffer size=%d bytes (~%.2fs duration)",
                        session.session_id,
                        current_turn,
                        len(audio_data),
                        duration_sec,
                    )

                    # Launch pipeline turn task
                    task = asyncio.create_task(
                        execute_turn(session, websocket, audio_data, current_turn)
                    )
                    session.current_task = task

                # 3. Text fallback for testing or manual input
                elif event_type == "text_input":
                    user_text = str(payload.get("text", "")).strip()
                    if not user_text:
                        continue

                    session.turn_id += 1
                    current_turn = session.turn_id
                    logger.info("[Session %s | Turn %d] Received manual text input: %r", session.session_id, current_turn, user_text)

                    async def _run_text_turn(t_id: int, text: str) -> None:
                        async with session.turn_lock:
                            try:
                                await websocket.send_json({
                                    "type": "state",
                                    "state": "processing",
                                    "turn_id": t_id,
                                })
                                await websocket.send_json({
                                    "type": "transcript",
                                    "role": "user",
                                    "text": text,
                                    "turn_id": t_id,
                                })
                                logger.info("[Session %s | Turn %d] Text turn: submitting to Qwen: %r", session.session_id, t_id, text)
                                reply = await asyncio.to_thread(
                                    generate_voice_response,
                                    list(session.history),
                                    text,
                                    session.system_prompt,
                                )
                                if not reply or not reply.strip():
                                    reply = "I understand. How else can I help you today?"

                                session.add_turn(text, reply)
                                logger.info("[Session %s | Turn %d] Text turn: Qwen response: %r", session.session_id, t_id, reply)

                                await websocket.send_json({
                                    "type": "transcript",
                                    "role": "assistant",
                                    "text": reply,
                                    "turn_id": t_id,
                                })

                                wav = await asyncio.to_thread(synthesize_text_to_wav, reply)
                                if wav:
                                    session.is_ai_speaking = True
                                    logger.info("[Session %s | Turn %d] Text turn: delivering %d bytes WAV audio", session.session_id, t_id, len(wav))
                                    await websocket.send_json({
                                        "type": "ai_audio_start",
                                        "turn_id": t_id,
                                        "format": "wav",
                                        "byte_length": len(wav),
                                    })
                                    await websocket.send_bytes(wav)
                                    await websocket.send_json({
                                        "type": "state",
                                        "state": "speaking",
                                        "turn_id": t_id,
                                    })
                                else:
                                    await websocket.send_json({
                                        "type": "state",
                                        "state": "listening",
                                        "turn_id": t_id,
                                    })
                            except Exception as exc:
                                logger.exception("[Session %s | Turn %d] Text turn error: %s", session.session_id, t_id, exc)

                    session.current_task = asyncio.create_task(_run_text_turn(current_turn, user_text))

                # 4. Audio playback finished on frontend
                elif event_type == "audio_playback_ended":
                    session.is_ai_speaking = False
                    logger.info("[Session %s | Turn %d] Frontend completed audio playback", session.session_id, session.turn_id)
                    await websocket.send_json({
                        "type": "state",
                        "state": "listening",
                        "turn_id": session.turn_id,
                    })

                # 5. Clear conversation history
                elif event_type == "clear_history":
                    session.history.clear()
                    logger.info("[Session %s] Cleared conversation history", session.session_id)
                    await websocket.send_json({
                        "type": "history_cleared",
                        "session_id": session.session_id,
                    })

                # 6. Update user identity & assistant personalization
                elif event_type == "update_profile":
                    fn = str(payload.get("first_name", "")).strip()
                    ln = str(payload.get("last_name", "")).strip()
                    an = str(payload.get("assistant_name", "")).strip()
                    if not session.user_profile:
                        session.user_profile = {}
                    session.user_profile["first_name"] = fn
                    session.user_profile["last_name"] = ln
                    session.user_profile["assistant_name"] = an
                    session.system_prompt = build_system_prompt(session.user_profile, session.dynamic_context)
                    logger.info(
                        "[Session %s] Profile updated via message: User='%s %s', Assistant='%s'",
                        session.session_id,
                        fn,
                        ln,
                        an,
                    )

                    # Persist to Supabase if authenticated with valid UUID
                    if is_valid_uuid(session.user_id):
                        try:
                            client = get_supabase_client(use_service_role=True)
                            client.table("users").update({
                                "first_name": fn,
                                "last_name": ln,
                                "assistant_name": an,
                            }).eq("user_id", session.user_id).execute()
                            logger.info("[Session %s] Persisted profile update to Supabase for user %s", session.session_id, session.user_id)
                        except Exception as exc:
                            logger.warning("[Session %s] Could not persist profile update to Supabase: %s", session.session_id, exc)

                    await websocket.send_json({
                        "type": "profile_updated",
                        "first_name": fn,
                        "last_name": ln,
                        "assistant_name": an,
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session.session_id)
    except Exception as exc:
        logger.exception("WebSocket connection error: %s", exc)
    finally:
        session.cancel_active_task()
