import React, { useState, useEffect, useRef } from "react";
import {
  Phone,
  PhoneCall,
  PhoneOff,
  Mic,
  MicOff,
  Sparkles,
  Radio,
  User,
  ShieldCheck,
  AlertCircle,
  Clock,
  Volume2,
} from "lucide-react";

const SPEECH_ENERGY_THRESHOLD = 0.015;
const BARGE_IN_THRESHOLD = 0.035;
const SILENCE_TIMEOUT_MS = 1400;

export default function CallerInterface({
  token,
  ownerFirstName = "Charles",
  ownerLastName = "Smith",
  assistantName = "Alaana",
}) {
  const [callerNumber, setCallerNumber] = useState("9824378001");
  const [isCalling, setIsCalling] = useState(false);
  const [callMode, setCallMode] = useState("IDLE"); // IDLE, RINGING, AI_SCREENING, HOLD_AND_WHISPER, AUDIO_BRIDGE, DECLINED, ENDED
  const [callStatusMsg, setCallStatusMsg] = useState("");
  const [callDuration, setCallDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [history, setHistory] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");

  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const activeAudioSourceRef = useRef(null);
  const durationTimerRef = useRef(null);
  const chatBottomRef = useRef(null);

  const isUserSpeakingRef = useRef(false);
  const silenceTimerRef = useRef(null);
  const isMutedRef = useRef(false);
  const isAISpeakingRef = useRef(false);
  const callModeRef = useRef("IDLE");

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    callModeRef.current = callMode;
  }, [callMode]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  useEffect(() => {
    if (isCalling && callMode !== "IDLE" && callMode !== "ENDED") {
      durationTimerRef.current = setInterval(() => {
        setCallDuration((prev) => prev + 1);
      }, 1000);
    } else {
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
    }
    return () => {
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
    };
  }, [isCalling, callMode]);

  const formatDuration = (sec) => {
    const mins = Math.floor(sec / 60);
    const secs = sec % 60;
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  };

  const stopAudioPlayback = () => {
    if (activeAudioSourceRef.current) {
      try {
        activeAudioSourceRef.current.stop();
      } catch (err) {}
      activeAudioSourceRef.current = null;
    }
    isAISpeakingRef.current = false;
  };

  const playWavAudioBytes = async (arrayBuffer) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }

      stopAudioPlayback();

      const decodedBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      const source = audioContextRef.current.createBufferSource();
      source.buffer = decodedBuffer;
      source.connect(audioContextRef.current.destination);

      activeAudioSourceRef.current = source;
      isAISpeakingRef.current = true;

      source.onended = () => {
        if (activeAudioSourceRef.current === source) {
          activeAudioSourceRef.current = null;
          isAISpeakingRef.current = false;
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "audio_playback_ended" }));
          }
        }
      };

      source.start(0);
    } catch (err) {
      console.error("Audio playback error:", err);
    }
  };

  const startCall = async () => {
    setErrorMsg("");
    setHistory([]);
    setCallDuration(0);

    try {
      // 1. Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
        },
      });
      mediaStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (isMutedRef.current) {
          setVolumeLevel(0);
          return;
        }

        const inputData = e.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        setVolumeLevel(Math.min(1, rms * 10));

        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcm16[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7fff;
        }

        // If in direct AUDIO_BRIDGE mode -> Stream raw audio chunks directly to owner
        if (callModeRef.current === "AUDIO_BRIDGE") {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(pcm16.buffer);
          }
          return;
        }

        // In AI_SCREENING mode: apply VAD
        const isAISpeaking = isAISpeakingRef.current;
        const threshold = isAISpeaking ? BARGE_IN_THRESHOLD : SPEECH_ENERGY_THRESHOLD;
        const isAbove = rms > threshold;

        if (isAISpeaking && rms > BARGE_IN_THRESHOLD) {
          stopAudioPlayback();
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "interrupt" }));
          }
        }

        if (isAbove) {
          if (!isUserSpeakingRef.current) {
            isUserSpeakingRef.current = true;
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: "user_speaking" }));
            }
          }
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
          }
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(pcm16.buffer);
          }
        } else if (isUserSpeakingRef.current) {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(pcm16.buffer);
          }
          if (!silenceTimerRef.current) {
            silenceTimerRef.current = setTimeout(() => {
              isUserSpeakingRef.current = false;
              silenceTimerRef.current = null;
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: "speech_end" }));
              }
            }, SILENCE_TIMEOUT_MS);
          }
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      // 2. Connect to Call Hub WebSocket
      const host = window.location.hostname || "127.0.0.1";
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const port = window.location.port === "5173" ? "" : ":8000";
      const wsUrl = `${wsProtocol}//${host}${port || (window.location.port ? `:${window.location.port}` : "")}/ws/call-hub?role=caller&caller_phone_number=${encodeURIComponent(
        callerNumber
      )}&first_name=${encodeURIComponent(ownerFirstName)}&last_name=${encodeURIComponent(
        ownerLastName
      )}&assistant_name=${encodeURIComponent(assistantName)}`;

      console.log("Caller connecting to:", wsUrl);
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      setIsCalling(true);
      setCallMode("RINGING");
      setCallStatusMsg("Calling...");

      ws.onmessage = async (event) => {
        // Binary WAV audio chunk from server (AI speech or forwarded Owner speech)
        if (event.data instanceof ArrayBuffer) {
          await playWavAudioBytes(event.data);
          return;
        }

        try {
          const msg = JSON.parse(event.data);
          console.log("[Caller WS]", msg);

          if (msg.type === "call_ringing") {
            setCallMode("RINGING");
            setCallStatusMsg(msg.message || "Ringing...");
          } else if (msg.type === "call_connected") {
            setCallMode(msg.mode || "AI_SCREENING");
            setCallStatusMsg(`Connected with ${msg.assistant_name || "Assistant"}`);
          } else if (msg.type === "call_mode") {
            setCallMode(msg.mode);
            if (msg.message) setCallStatusMsg(msg.message);
          } else if (msg.type === "call_transferred") {
            setCallMode("AUDIO_BRIDGE");
            setCallStatusMsg(msg.message || `Connected directly with ${ownerFirstName}!`);
          } else if (msg.type === "call_declined") {
            setCallMode("DECLINED");
            setCallStatusMsg(msg.message || "Call was declined.");
          } else if (msg.type === "call_ended") {
            setCallMode("ENDED");
            setCallStatusMsg(msg.reason || "Call ended.");
            setTimeout(cleanupCall, 2000);
          } else if (msg.type === "transcript") {
            setHistory((prev) => {
              const idx = prev.findIndex((m) => m.turnId === msg.turn_id && m.role === msg.role);
              if (idx !== -1) {
                const copy = [...prev];
                copy[idx] = { role: msg.role, content: msg.text, turnId: msg.turn_id };
                return copy;
              }
              return [...prev, { role: msg.role, content: msg.text, turnId: msg.turn_id }];
            });
          }
        } catch (err) {
          console.error("Caller JSON parse error:", err);
        }
      };

      ws.onerror = (err) => {
        console.error("Caller WS error:", err);
        setErrorMsg("Connection error to call hub. Ensure backend is running.");
      };

      ws.onclose = () => {
        cleanupCall();
      };
    } catch (err) {
      console.error("Start call error:", err);
      setErrorMsg(`Failed to start call: ${err.message}. Did you allow microphone access?`);
      cleanupCall();
    }
  };

  const endCall = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "end_call" }));
    }
    cleanupCall();
  };

  const cleanupCall = () => {
    stopAudioPlayback();
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) wsRef.current.close();
      wsRef.current = null;
    }
    setIsCalling(false);
    setCallMode("IDLE");
    setVolumeLevel(0);
  };

  const getStatusBadge = () => {
    switch (callMode) {
      case "RINGING":
        return { label: "Ringing...", color: "bg-amber-500/20 text-amber-400 border-amber-500/40" };
      case "AI_SCREENING":
        return { label: `Speaking with ${assistantName} (AI)`, color: "bg-blue-500/20 text-blue-400 border-blue-500/40" };
      case "HOLD_AND_WHISPER":
        return { label: "On Hold (Connecting to Owner)", color: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40" };
      case "AUDIO_BRIDGE":
        return { label: `Connected with ${ownerFirstName} (Live Audio Bridge)`, color: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" };
      case "DECLINED":
        return { label: "Call Declined", color: "bg-rose-500/20 text-rose-400 border-rose-500/40" };
      case "ENDED":
        return { label: "Call Ended", color: "bg-slate-700/40 text-slate-400 border-slate-600/40" };
      default:
        return { label: "Ready to Call", color: "bg-slate-700/40 text-slate-400 border-slate-600/40" };
    }
  };

  const badge = getStatusBadge();

  return (
    <div style={{ maxWidth: "480px", margin: "0 auto", padding: "16px", display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Mobile Phone Mock Container */}
      <div
        className="glass-panel"
        style={{
          borderRadius: "28px",
          padding: "24px 20px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "20px",
          boxShadow: "0 10px 40px rgba(0,0,0,0.4)",
          position: "relative",
        }}
      >
        {/* Top Phone Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Phone size={18} color="#10b981" />
            <span style={{ fontSize: "0.9rem", fontWeight: "700" }}>Mobile Caller Device</span>
          </div>
          {isCalling && (
            <div style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "0.8rem", color: "#34d399" }}>
              <Clock size={14} />
              <span>{formatDuration(callDuration)}</span>
            </div>
          )}
        </div>

        {errorMsg && (
          <div style={{ padding: "10px", borderRadius: "8px", background: "rgba(244,63,94,0.15)", color: "#fecdd3", fontSize: "0.8rem", width: "100%" }}>
            {errorMsg}
          </div>
        )}

        {/* Recipient Card */}
        <div style={{ textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
          <div
            style={{
              width: "72px",
              height: "72px",
              borderRadius: "50%",
              background: callMode === "AUDIO_BRIDGE" ? "linear-gradient(135deg, #10b981, #059669)" : "linear-gradient(135deg, #3b82f6, #8b5cf6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 24px rgba(59, 130, 246, 0.4)",
            }}
          >
            <User size={36} color="#ffffff" />
          </div>
          <h2 style={{ fontSize: "1.3rem", fontWeight: "700", margin: 0 }}>
            {ownerFirstName} {ownerLastName}
          </h2>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", margin: 0 }}>
            Personal Assistant: {assistantName}
          </p>
        </div>

        {/* Call Mode Status Badge */}
        <div style={{ padding: "6px 16px", borderRadius: "20px", border: "1px solid", fontSize: "0.8rem", fontWeight: "600" }} className={badge.color}>
          {badge.label}
        </div>

        {/* Caller Number Selector / Input (when idle) */}
        {!isCalling ? (
          <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: "8px" }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Simulated Caller Phone Number:</label>
            <input
              type="text"
              value={callerNumber}
              onChange={(e) => setCallerNumber(e.target.value)}
              placeholder="e.g. 9824378001 or 6356414812"
              style={{
                width: "100%",
                padding: "10px 14px",
                borderRadius: "10px",
                background: "rgba(0,0,0,0.3)",
                border: "1px solid var(--border-color)",
                color: "#f8fafc",
                fontSize: "1rem",
                textAlign: "center",
                letterSpacing: "0.05em",
              }}
            />
            <div style={{ display: "flex", gap: "6px", justifyContent: "center" }}>
              <button
                onClick={() => setCallerNumber("6356414812")}
                style={{ fontSize: "0.7rem", padding: "4px 8px", borderRadius: "6px", background: "rgba(245, 158, 11, 0.15)", border: "1px solid rgba(245,158,11,0.3)", color: "#f59e0b", cursor: "pointer" }}
                title="Priority Contact (Routes directly to owner)"
              >
                ★ Priority Number (6356414812)
              </button>
              <button
                onClick={() => setCallerNumber("9824378001")}
                style={{ fontSize: "0.7rem", padding: "4px 8px", borderRadius: "6px", background: "rgba(59, 130, 246, 0.15)", border: "1px solid rgba(59,130,246,0.3)", color: "#60a5fa", cursor: "pointer" }}
                title="Known Contact (Evaluated by AI)"
              >
                Regular Contact (9824378001)
              </button>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: "0.85rem", color: "#93c5fd" }}>
            Calling from: <strong>{callerNumber}</strong>
          </div>
        )}

        {/* Audio Wave Visualizer */}
        <div style={{ height: "40px", display: "flex", alignItems: "center", justifyContent: "center", gap: "4px", width: "100%" }}>
          {isCalling &&
            Array.from({ length: 18 }).map((_, i) => (
              <div
                key={i}
                style={{
                  width: "4px",
                  height: `${Math.min(36, Math.max(4, volumeLevel * 40 * Math.sin((i / 18) * Math.PI)))}px`,
                  borderRadius: "2px",
                  backgroundColor: callMode === "AUDIO_BRIDGE" ? "#10b981" : "#3b82f6",
                  transition: "height 0.08s ease",
                }}
              />
            ))}
        </div>

        {/* Call Controls Button Bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "16px", marginTop: "10px" }}>
          {!isCalling ? (
            <button
              onClick={startCall}
              style={{
                width: "68px",
                height: "68px",
                borderRadius: "50%",
                background: "linear-gradient(135deg, #10b981, #059669)",
                border: "none",
                color: "#ffffff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                boxShadow: "0 6px 20px rgba(16, 185, 129, 0.4)",
              }}
              title="Call"
            >
              <PhoneCall size={28} />
            </button>
          ) : (
            <>
              {/* Mute button */}
              <button
                onClick={() => setIsMuted(!isMuted)}
                style={{
                  width: "52px",
                  height: "52px",
                  borderRadius: "50%",
                  background: isMuted ? "rgba(244, 63, 94, 0.2)" : "rgba(255, 255, 255, 0.08)",
                  border: `1px solid ${isMuted ? "rgba(244, 63, 94, 0.4)" : "var(--border-color)"}`,
                  color: isMuted ? "#fb7185" : "#f8fafc",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                }}
                title={isMuted ? "Unmute" : "Mute"}
              >
                {isMuted ? <MicOff size={20} /> : <Mic size={20} />}
              </button>

              {/* End Call button */}
              <button
                onClick={endCall}
                style={{
                  width: "68px",
                  height: "68px",
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, #e11d48, #be123c)",
                  border: "none",
                  color: "#ffffff",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                  boxShadow: "0 6px 20px rgba(225, 29, 72, 0.4)",
                }}
                title="End Call"
              >
                <PhoneOff size={28} />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Transcript Box */}
      <div
        className="glass-panel"
        style={{
          borderRadius: "20px",
          padding: "16px",
          maxHeight: "320px",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.8rem", color: "var(--text-muted)" }}>
          <Sparkles size={14} color="#3b82f6" />
          <span>Call Transcript</span>
        </div>
        {history.length === 0 ? (
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", textAlign: "center", padding: "16px 0" }}>
            Start the call to speak. When connected with the AI receptionist or owner, transcript will appear here.
          </p>
        ) : (
          history.map((msg, idx) => {
            const isMe = msg.role === "user";
            return (
              <div
                key={idx}
                style={{
                  alignSelf: isMe ? "flex-end" : "flex-start",
                  maxWidth: "85%",
                  background: isMe ? "linear-gradient(135deg, #2563eb, #1d4ed8)" : "rgba(30, 41, 59, 0.8)",
                  padding: "10px 14px",
                  borderRadius: isMe ? "14px 14px 2px 14px" : "14px 14px 14px 2px",
                  color: "#ffffff",
                  fontSize: "0.85rem",
                  lineHeight: "1.4",
                }}
              >
                {msg.content}
              </div>
            );
          })
        )}
        <div ref={chatBottomRef} />
      </div>
    </div>
  );
}
