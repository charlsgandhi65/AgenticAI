import React, { useState, useEffect, useRef } from "react";
import {
  Phone,
  PhoneCall,
  PhoneOff,
  PhoneForwarded,
  ShieldAlert,
  ShieldCheck,
  User,
  Sparkles,
  Radio,
  Check,
  X,
  Mic,
  MicOff,
  AlertTriangle,
  Volume2,
} from "lucide-react";

export default function OwnerDashboard({
  token,
  ownerFirstName = "Charles",
  ownerLastName = "Smith",
  assistantName = "Alaana",
}) {
  const [isConnected, setIsConnected] = useState(false);
  const [activeCall, setActiveCall] = useState(null);
  const [whisperAlert, setWhisperAlert] = useState(null);
  const [isBridgeActive, setIsBridgeActive] = useState(false);
  const [isOwnerMuted, setIsOwnerMuted] = useState(false);
  const [history, setHistory] = useState([]);
  const [statusMsg, setStatusMsg] = useState("Waiting for incoming calls...");

  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const activeAudioSourceRef = useRef(null);
  const chatBottomRef = useRef(null);

  const isOwnerMutedRef = useRef(false);
  const isBridgeActiveRef = useRef(false);

  useEffect(() => {
    isOwnerMutedRef.current = isOwnerMuted;
  }, [isOwnerMuted]);

  useEffect(() => {
    isBridgeActiveRef.current = isBridgeActive;
  }, [isBridgeActive]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  // Connect WebSocket to Call Hub as Owner
  useEffect(() => {
    const host = window.location.hostname || "127.0.0.1";
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const port = window.location.port === "5173" ? "" : ":8000";
    const wsUrl = `${wsProtocol}//${host}${port || (window.location.port ? `:${window.location.port}` : "")}/ws/call-hub?role=owner&first_name=${encodeURIComponent(
      ownerFirstName
    )}&last_name=${encodeURIComponent(ownerLastName)}&assistant_name=${encodeURIComponent(assistantName)}${
      token ? `&token=${encodeURIComponent(token)}` : ""
    }`;

    console.log("Owner connecting to:", wsUrl);
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setStatusMsg("Online • Monitoring incoming calls");
    };

    ws.onmessage = async (event) => {
      // In AUDIO_BRIDGE mode, incoming binary is Caller's voice audio!
      if (event.data instanceof ArrayBuffer) {
        await playBridgeAudioBytes(event.data);
        return;
      }

      try {
        const msg = JSON.parse(event.data);
        console.log("[Owner WS]", msg);

        if (msg.type === "owner_ready") {
          setIsConnected(true);
        } else if (msg.type === "incoming_call") {
          setActiveCall({
            callId: msg.call_id,
            callerNumber: msg.caller_number,
            callerName: msg.caller_name || "Unknown Caller",
            callerType: msg.caller_type || "UNKNOWN",
            category: msg.category || "",
            decision: msg.decision || "HANDLE",
            decisionReason: msg.decision_reason || "",
            isPriority: msg.is_priority || false,
            requiresTakeover: msg.requires_takeover || false,
            aiHandling: msg.ai_handling || false,
          });
          setStatusMsg(msg.message || `Incoming call from ${msg.caller_name || msg.caller_number}`);
        } else if (msg.type === "whisper_escalation") {
          setWhisperAlert({
            callId: msg.call_id,
            callerNumber: msg.caller_number,
            callerName: msg.caller_name || "Caller",
            whisperText: msg.whisper_text,
          });
        } else if (msg.type === "bridge_connected") {
          setIsBridgeActive(true);
          setWhisperAlert(null);
          setStatusMsg("🟢 Connected! Live two-way audio bridge active with caller.");
          startOwnerMicrophoneBridge();
        } else if (msg.type === "transfer_rejected") {
          setWhisperAlert(null);
          setStatusMsg("Transfer declined. AI receptionist taking message.");
        } else if (msg.type === "call_ended") {
          setActiveCall(null);
          setWhisperAlert(null);
          setIsBridgeActive(false);
          stopOwnerMicrophoneBridge();
          setStatusMsg(msg.reason || "Call ended. Ready for next call.");
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
        console.error("Owner message parse error:", err);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      setStatusMsg("Disconnected from Call Hub");
      stopOwnerMicrophoneBridge();
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) ws.close();
      stopOwnerMicrophoneBridge();
    };
  }, [token, ownerFirstName, ownerLastName, assistantName]);

  // Play caller audio received in AUDIO_BRIDGE mode
  const playBridgeAudioBytes = async (arrayBuffer) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }
      const decodedBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      const source = audioContextRef.current.createBufferSource();
      source.buffer = decodedBuffer;
      source.connect(audioContextRef.current.destination);
      source.start(0);
    } catch (err) {
      console.error("Bridge audio play error:", err);
    }
  };

  // Start capturing Owner Laptop Microphone to stream to Caller
  const startOwnerMicrophoneBridge = async () => {
    try {
      if (mediaStreamRef.current) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
      mediaStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (isOwnerMutedRef.current || !isBridgeActiveRef.current) return;
        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcm16[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7fff;
        }
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(pcm16.buffer);
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
    } catch (err) {
      console.error("Owner mic error:", err);
    }
  };

  const stopOwnerMicrophoneBridge = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  };

  // Accept Transfer / Take Over
  const handleAcceptTransfer = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "accept_transfer" }));
    }
  };

  // Decline Transfer
  const handleRejectTransfer = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reject_transfer" }));
    }
    setWhisperAlert(null);
  };

  // End Call
  const handleEndCall = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "end_call" }));
    }
    setActiveCall(null);
    setWhisperAlert(null);
    setIsBridgeActive(false);
    stopOwnerMicrophoneBridge();
  };

  const getCallerTypeBadge = (type) => {
    switch (type) {
      case "PRIORITY_CONTACT":
        return { label: "★ PRIORITY CONTACT", color: "bg-amber-500/20 text-amber-300 border-amber-500/50" };
      case "KNOWN_CONTACT":
        return { label: "KNOWN CONTACT", color: "bg-blue-500/20 text-blue-300 border-blue-500/50" };
      case "SPAM":
        return { label: "SPAM CACHE", color: "bg-rose-500/20 text-rose-300 border-rose-500/50" };
      default:
        return { label: "UNKNOWN CALLER", color: "bg-slate-700/40 text-slate-300 border-slate-600/50" };
    }
  };

  const getDecisionBadge = (decision) => {
    switch (decision) {
      case "HANDLE":
        return { label: "ACTION: HANDLE (AI Receptionist)", color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40" };
      case "DECLINE":
        return { label: "ACTION: DECLINE", color: "bg-rose-500/20 text-rose-400 border-rose-500/40" };
      case "ESCALATE":
        return { label: "ACTION: ESCALATE (Direct to Owner)", color: "bg-purple-500/20 text-purple-300 border-purple-500/40" };
      default:
        return { label: decision, color: "bg-slate-700/40 text-slate-400 border-slate-600/40" };
    }
  };

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto", padding: "16px", display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Top Banner Status */}
      <div
        className="glass-panel"
        style={{
          padding: "16px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              width: "42px",
              height: "42px",
              borderRadius: "12px",
              background: "linear-gradient(135deg, #8b5cf6, #6d28d9)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 20px rgba(139, 92, 246, 0.4)",
            }}
          >
            <Radio size={22} color="#ffffff" />
          </div>
          <div>
            <h2 style={{ fontSize: "1.15rem", fontWeight: "700", margin: 0 }}>
              Owner Call Station ({ownerFirstName} {ownerLastName})
            </h2>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: 0 }}>
              AI Receptionist: {assistantName} • Status: {statusMsg}
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "6px 12px",
              borderRadius: "8px",
              background: isConnected ? "rgba(16, 185, 129, 0.15)" : "rgba(244, 63, 94, 0.15)",
              border: `1px solid ${isConnected ? "rgba(16, 185, 129, 0.3)" : "rgba(244, 63, 94, 0.3)"}`,
              fontSize: "0.8rem",
              color: isConnected ? "#34d399" : "#fb7185",
            }}
          >
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: isConnected ? "#34d399" : "#fb7185" }} />
            {isConnected ? "Hub Connected" : "Connecting..."}
          </div>
        </div>
      </div>

      {/* 1. PRIORITY CONTACT DIRECT RINGING BANNER */}
      {activeCall && activeCall.isPriority && (
        <div
          style={{
            padding: "18px 24px",
            borderRadius: "16px",
            background: "linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(217, 119, 6, 0.15))",
            border: "1px solid rgba(245, 158, 11, 0.6)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "14px",
            boxShadow: "0 4px 24px rgba(245, 158, 11, 0.25)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
            <div style={{ width: "48px", height: "48px", borderRadius: "50%", background: "#f59e0b", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <PhoneCall size={26} color="#ffffff" className="animate-pulse" />
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontWeight: "700", fontSize: "1.1rem", color: "#fef3c7" }}>
                  ★ PRIORITY CALL: {activeCall.callerName}
                </span>
                <span style={{ fontSize: "0.75rem", padding: "2px 8px", borderRadius: "6px", background: "rgba(0,0,0,0.3)", color: "#fde68a" }}>
                  {activeCall.category || "VIP"}
                </span>
              </div>
              <p style={{ fontSize: "0.85rem", color: "#fef3c7", margin: "4px 0 0 0" }}>
                Priority contact routed directly to you without AI intervention. Number: {activeCall.callerNumber}
              </p>
            </div>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <button
              onClick={handleAcceptTransfer}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                background: "linear-gradient(135deg, #10b981, #059669)",
                color: "#ffffff",
                border: "none",
                borderRadius: "10px",
                padding: "10px 20px",
                fontSize: "0.9rem",
                fontWeight: "600",
                cursor: "pointer",
                boxShadow: "0 4px 16px rgba(16, 185, 129, 0.4)",
              }}
            >
              <PhoneCall size={18} />
              <span>Accept & Speak Now</span>
            </button>
            <button
              onClick={handleEndCall}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                background: "rgba(244, 63, 94, 0.2)",
                border: "1px solid rgba(244, 63, 94, 0.4)",
                color: "#fda4af",
                borderRadius: "10px",
                padding: "10px 16px",
                fontSize: "0.9rem",
                cursor: "pointer",
              }}
            >
              <PhoneOff size={16} />
              <span>Decline</span>
            </button>
          </div>
        </div>
      )}

      {/* 2. PRIVATE WHISPER & ESCALATION ALERT */}
      {whisperAlert && (
        <div
          style={{
            padding: "18px 24px",
            borderRadius: "16px",
            background: "linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(109, 40, 217, 0.15))",
            border: "1px solid rgba(139, 92, 246, 0.6)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "14px",
            boxShadow: "0 4px 24px rgba(139, 92, 246, 0.3)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "14px", flex: 1, minWidth: "260px" }}>
            <div style={{ width: "46px", height: "46px", borderRadius: "50%", background: "#8b5cf6", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <PhoneForwarded size={22} color="#ffffff" />
            </div>
            <div>
              <div style={{ fontWeight: "700", fontSize: "1rem", color: "#e9d5ff" }}>
                🔒 Private Whisper: Call Escalation Requested
              </div>
              <p style={{ fontSize: "0.85rem", color: "#f3e8ff", margin: "4px 0 0 0", lineHeight: "1.4" }}>
                {whisperAlert.whisperText}
              </p>
            </div>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <button
              onClick={handleAcceptTransfer}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                background: "linear-gradient(135deg, #10b981, #059669)",
                color: "#ffffff",
                border: "none",
                borderRadius: "10px",
                padding: "10px 18px",
                fontSize: "0.9rem",
                fontWeight: "600",
                cursor: "pointer",
                boxShadow: "0 4px 14px rgba(16, 185, 129, 0.4)",
              }}
            >
              <Check size={18} />
              <span>Accept Takeover</span>
            </button>
            <button
              onClick={handleRejectTransfer}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                background: "rgba(244, 63, 94, 0.2)",
                border: "1px solid rgba(244, 63, 94, 0.4)",
                color: "#fda4af",
                borderRadius: "10px",
                padding: "10px 16px",
                fontSize: "0.9rem",
                cursor: "pointer",
              }}
            >
              <X size={16} />
              <span>AI Take Message</span>
            </button>
          </div>
        </div>
      )}

      {/* 3. ACTIVE TWO-WAY AUDIO BRIDGE BAR */}
      {isBridgeActive && (
        <div
          style={{
            padding: "16px 20px",
            borderRadius: "14px",
            background: "linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.15))",
            border: "1px solid rgba(16, 185, 129, 0.5)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "12px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#34d399" }} className="animate-ping" />
            <span style={{ fontWeight: "600", color: "#a7f3d0", fontSize: "0.9rem" }}>
              Live Two-Way Audio Bridge Active (You & Caller)
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <button
              onClick={() => setIsOwnerMuted(!isOwnerMuted)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "8px 14px",
                borderRadius: "8px",
                background: isOwnerMuted ? "rgba(244, 63, 94, 0.25)" : "rgba(255, 255, 255, 0.1)",
                border: `1px solid ${isOwnerMuted ? "rgba(244, 63, 94, 0.5)" : "var(--border-color)"}`,
                color: isOwnerMuted ? "#fb7185" : "#f8fafc",
                fontSize: "0.85rem",
                cursor: "pointer",
              }}
            >
              {isOwnerMuted ? <MicOff size={16} /> : <Mic size={16} />}
              <span>{isOwnerMuted ? "Unmute My Mic" : "Mute My Mic"}</span>
            </button>
            <button
              onClick={handleEndCall}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "8px 16px",
                borderRadius: "8px",
                background: "linear-gradient(135deg, #e11d48, #be123c)",
                border: "none",
                color: "#ffffff",
                fontSize: "0.85rem",
                fontWeight: "600",
                cursor: "pointer",
              }}
            >
              <PhoneOff size={16} />
              <span>Hang Up Call</span>
            </button>
          </div>
        </div>
      )}

      {/* 4. ACTIVE CALL METADATA & DECISION ENGINE CARD */}
      {activeCall && (
        <div
          className="glass-panel"
          style={{
            padding: "20px 24px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "16px",
          }}
        >
          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "4px" }}>Caller Identity</div>
            <div style={{ fontWeight: "700", fontSize: "1.05rem" }}>{activeCall.callerName}</div>
            <div style={{ fontSize: "0.85rem", color: "#93c5fd" }}>{activeCall.callerNumber}</div>
          </div>

          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "4px" }}>Caller Classification</div>
            <div style={{ display: "inline-block", padding: "4px 10px", borderRadius: "8px", border: "1px solid", fontSize: "0.8rem", fontWeight: "600" }} className={getCallerTypeBadge(activeCall.callerType).color}>
              {getCallerTypeBadge(activeCall.callerType).label}
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "4px" }}>Decision Engine Status</div>
            <div style={{ display: "inline-block", padding: "4px 10px", borderRadius: "8px", border: "1px solid", fontSize: "0.8rem", fontWeight: "600" }} className={getDecisionBadge(activeCall.decision).color}>
              {getDecisionBadge(activeCall.decision).label}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
              {activeCall.decisionReason}
            </div>
          </div>
        </div>
      )}

      {/* 5. LIVE CALL CONVERSATION TRANSCRIPT STREAM */}
      <div
        className="glass-panel"
        style={{
          padding: "20px",
          borderRadius: "16px",
          display: "flex",
          flexDirection: "column",
          gap: "14px",
          minHeight: "380px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid var(--border-color)", paddingBottom: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Sparkles size={16} color="#8b5cf6" />
            <span style={{ fontSize: "0.9rem", fontWeight: "600" }}>Live Conversation Monitoring (Owner Screen)</span>
          </div>
          {activeCall && (
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Call ID: {activeCall.callId}
            </span>
          )}
        </div>

        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "12px", minHeight: "260px" }}>
          {history.length === 0 ? (
            <div style={{ margin: "auto", textAlign: "center", color: "var(--text-muted)", padding: "40px 20px" }}>
              <Volume2 size={36} style={{ margin: "0 auto 10px auto", opacity: 0.3 }} />
              <p style={{ fontSize: "0.95rem", fontWeight: "500", margin: 0 }}>No active speech stream</p>
              <p style={{ fontSize: "0.8rem", margin: "4px 0 0 0" }}>
                When a call arrives from the mobile device, caller speech and AI responses will stream here in real-time.
              </p>
            </div>
          ) : (
            history.map((msg, idx) => {
              const isCaller = msg.role === "user";
              return (
                <div
                  key={idx}
                  style={{
                    alignSelf: isCaller ? "flex-start" : "flex-end",
                    maxWidth: "80%",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: isCaller ? "flex-start" : "flex-end",
                  }}
                >
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "4px" }}>
                    {isCaller ? `Caller (${activeCall?.callerName || "Phone"})` : `${assistantName} (AI Receptionist)`}
                  </span>
                  <div
                    style={{
                      padding: "10px 16px",
                      borderRadius: isCaller ? "14px 14px 14px 2px" : "14px 14px 2px 14px",
                      background: isCaller ? "rgba(30, 41, 59, 0.9)" : "linear-gradient(135deg, #7c3aed, #6d28d9)",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                      color: "#ffffff",
                      fontSize: "0.9rem",
                      lineHeight: "1.45",
                    }}
                  >
                    {msg.content}
                  </div>
                </div>
              );
            })
          )}
          <div ref={chatBottomRef} />
        </div>
      </div>
    </div>
  );
}
