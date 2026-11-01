import React, { useState, useEffect, useRef } from "react";
import {
  Mic,
  MicOff,
  PhoneCall,
  PhoneOff,
  Square,
  Sparkles,
  Volume2,
  VolumeX,
  Radio,
  Send,
  Trash2,
  Key,
  ShieldCheck,
  AlertCircle,
  User,
  Check,
  Settings,
} from "lucide-react";

import CallerInterface from "./CallerInterface";
import OwnerDashboard from "./OwnerDashboard";

const SPEECH_ENERGY_THRESHOLD = 0.015; // RMS threshold to detect human speech
const BARGE_IN_THRESHOLD = 0.035; // Higher threshold when AI is speaking to avoid speaker acoustic echo
const SILENCE_TIMEOUT_MS = 1400; // Milliseconds of silence to trigger end-of-speech (allows natural speech pauses)

export default function App() {
  // Navigation & Role State: "owner" | "caller" | "voice_test"
  const getInitialView = () => {
    const path = window.location.pathname.toLowerCase();
    const search = new URLSearchParams(window.location.search);
    const roleParam = search.get("role")?.toLowerCase();
    if (path.includes("caller") || roleParam === "caller") return "caller";
    if (path.includes("test") || roleParam === "test") return "voice_test";
    return "owner";
  };
  const [activeTab, setActiveTab] = useState(getInitialView);

  // Session & Connection State
  const [isConnected, setIsConnected] = useState(false);
  const [isCalling, setIsCalling] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [token, setToken] = useState(() => localStorage.getItem("supabase_token") || "");
  const [showTokenInput, setShowTokenInput] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // User Identity & Assistant Customization
  const [firstName, setFirstName] = useState(() => localStorage.getItem("calling_agent_first_name") || "Charles");
  const [lastName, setLastName] = useState(() => localStorage.getItem("calling_agent_last_name") || "Smith");
  const [assistantName, setAssistantName] = useState(() => localStorage.getItem("calling_agent_assistant_name") || "Alaana");
  const [showProfileConfig, setShowProfileConfig] = useState(false);
  const [profileSavedFeedback, setProfileSavedFeedback] = useState(false);

  // Pipeline State: "idle" | "listening" | "processing" | "speaking"
  const [pipelineState, setPipelineState] = useState("idle");
  const [isMuted, setIsMuted] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0); // 0 to 1 for visualizer
  const [history, setHistory] = useState([]);
  const [manualText, setManualText] = useState("");

  // Refs for Web Audio & Streaming
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const activeAudioSourceRef = useRef(null);
  const chatBottomRef = useRef(null);

  // VAD tracking refs
  const isUserSpeakingRef = useRef(false);
  const silenceTimerRef = useRef(null);
  const isMutedRef = useRef(false);
  const isAISpeakingRef = useRef(false);
  const currentTurnIdRef = useRef(0);

  // Keep refs synced with state
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    isAISpeakingRef.current = pipelineState === "speaking";
  }, [pipelineState]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, pipelineState]);

  // Save token to localStorage
  const handleSaveToken = (val) => {
    setToken(val);
    localStorage.setItem("supabase_token", val);
  };

  // Save User & Assistant profile to localStorage and send update to backend
  const handleSaveProfile = (fn, ln, an) => {
    setFirstName(fn);
    setLastName(ln);
    setAssistantName(an);
    localStorage.setItem("calling_agent_first_name", fn);
    localStorage.setItem("calling_agent_last_name", ln);
    localStorage.setItem("calling_agent_assistant_name", an);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "update_profile",
          first_name: fn,
          last_name: ln,
          assistant_name: an,
        })
      );
    }
    setProfileSavedFeedback(true);
    setTimeout(() => setProfileSavedFeedback(false), 2000);
  };

  // Stop currently playing AI audio (Barge-in / Interrupt)
  const stopAIAudioPlayback = () => {
    if (activeAudioSourceRef.current) {
      try {
        activeAudioSourceRef.current.stop();
      } catch (err) {
        // Source may already be stopped
      }
      activeAudioSourceRef.current = null;
    }
    isAISpeakingRef.current = false;
  };

  // Play incoming WAV binary audio chunk via Web Audio API
  const playWavAudioBytes = async (arrayBuffer) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 22050,
        });
      }
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }

      // Stop any existing playback first
      stopAIAudioPlayback();

      const decodedBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      const source = audioContextRef.current.createBufferSource();
      source.buffer = decodedBuffer;
      source.connect(audioContextRef.current.destination);

      activeAudioSourceRef.current = source;
      isAISpeakingRef.current = true;
      setPipelineState("speaking");

      source.onended = () => {
        if (activeAudioSourceRef.current === source) {
          activeAudioSourceRef.current = null;
          isAISpeakingRef.current = false;
          setPipelineState("listening");
          // Notify backend that speech has concluded
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "audio_playback_ended" }));
          }
        }
      };

      source.start(0);
    } catch (err) {
      console.error("Audio playback error:", err);
      setPipelineState("listening");
    }
  };

  // Start Live Conversation
  const startConversation = async () => {
    setErrorMsg("");
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

      // 2. Set up AudioContext for microphone capture at 16kHz
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // Real-time audio process & VAD
      processor.onaudioprocess = (e) => {
        if (isMutedRef.current) {
          setVolumeLevel(0);
          return;
        }

        const inputData = e.inputBuffer.getChannelData(0);

        // Compute RMS (Volume Energy)
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        setVolumeLevel(Math.min(1, rms * 12));

        const isAISpeaking = isAISpeakingRef.current;
        const activeThreshold = isAISpeaking ? BARGE_IN_THRESHOLD : SPEECH_ENERGY_THRESHOLD;
        const isAboveThreshold = rms > activeThreshold;

        // Convert Float32 to 16-bit signed PCM
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcm16[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7fff;
        }

        // BARGE-IN: If user speaks clearly while AI is speaking, interrupt immediately!
        if (isAISpeaking && rms > BARGE_IN_THRESHOLD) {
          console.log("Barge-in: User spoke while AI was speaking (RMS:", rms, ")");
          stopAIAudioPlayback();
          setPipelineState("listening");
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "interrupt" }));
          }
        }

        if (isAboveThreshold) {
          // User is currently speaking
          if (!isUserSpeakingRef.current) {
            isUserSpeakingRef.current = true;
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: "user_speaking" }));
            }
          }

          // Reset silence timer
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
          }

          // Stream binary audio chunk to backend
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(pcm16.buffer);
          }
        } else if (isUserSpeakingRef.current) {
          // Stream trailing silence frame so speech ends cleanly
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(pcm16.buffer);
          }

          // Start silence timer to detect end of speech (1400ms allows natural conversational pauses)
          if (!silenceTimerRef.current) {
            silenceTimerRef.current = setTimeout(() => {
              isUserSpeakingRef.current = false;
              silenceTimerRef.current = null;
              console.log("VAD: Silence detected (1400ms). Sending speech_end.");
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: "speech_end" }));
              }
            }, SILENCE_TIMEOUT_MS);
          }
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      // 3. Connect WebSocket to FastAPI backend with user identity & assistant parameters
      const host = window.location.hostname || "127.0.0.1";
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const params = new URLSearchParams();
      if (token) params.set("token", token);
      if (firstName) params.set("first_name", firstName);
      if (lastName) params.set("last_name", lastName);
      if (assistantName) params.set("assistant_name", assistantName);
      const wsUrl = `${wsProtocol}//${host}:8000/ws/voice-conversation?${params.toString()}`;

      console.log("Connecting WebSocket to:", wsUrl);
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setIsCalling(true);
        setPipelineState("listening");
        console.log("WebSocket connected.");
      };

      ws.onmessage = async (event) => {
        // If binary message, it's AI speech WAV audio!
        if (event.data instanceof ArrayBuffer) {
          await playWavAudioBytes(event.data);
          return;
        }

        // Handle JSON control & transcript messages
        try {
          const msg = JSON.parse(event.data);
          console.log("WS message:", msg);

          if (msg.type === "session_ready") {
            setSessionId(msg.session_id);
            if (msg.first_name) {
              setFirstName(msg.first_name);
              localStorage.setItem("calling_agent_first_name", msg.first_name);
            }
            if (msg.last_name) {
              setLastName(msg.last_name);
              localStorage.setItem("calling_agent_last_name", msg.last_name);
            }
            if (msg.assistant_name) {
              setAssistantName(msg.assistant_name);
              localStorage.setItem("calling_agent_assistant_name", msg.assistant_name);
            }
            if (msg.history && Array.isArray(msg.history)) {
              setHistory(msg.history);
            }
            setPipelineState("listening");
          } else if (msg.type === "profile_updated") {
            if (msg.first_name) setFirstName(msg.first_name);
            if (msg.last_name) setLastName(msg.last_name);
            if (msg.assistant_name) setAssistantName(msg.assistant_name);
          } else if (msg.type === "state") {
            setPipelineState(msg.state);
          } else if (msg.type === "transcript") {
            // Deduplicate by turnId and role so repeated speech is updated rather than duplicated
            setHistory((prev) => {
              const existingIdx = prev.findIndex(
                (m) => m.turnId === msg.turn_id && m.role === msg.role
              );
              if (existingIdx !== -1) {
                const updated = [...prev];
                updated[existingIdx] = {
                  role: msg.role,
                  content: msg.text,
                  turnId: msg.turn_id,
                };
                return updated;
              }
              return [
                ...prev,
                { role: msg.role, content: msg.text, turnId: msg.turn_id },
              ];
            });
          } else if (msg.type === "interrupted") {
            console.log("Turn interrupted by backend:", msg.turn_id);
            stopAIAudioPlayback();
            setPipelineState("listening");
          } else if (msg.type === "history_cleared") {
            setHistory([]);
          } else if (msg.type === "error") {
            setErrorMsg(msg.message);
          }
        } catch (err) {
          console.error("Failed to parse WS text message:", err);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        setErrorMsg("WebSocket connection error. Is the backend running on port 8000?");
      };

      ws.onclose = () => {
        console.log("WebSocket closed.");
        cleanupSession();
      };
    } catch (err) {
      console.error("Start conversation error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setErrorMsg("Microphone permission was denied. Please allow microphone access to talk to the AI.");
      } else {
        setErrorMsg(`Failed to start conversation: ${err.message}`);
      }
      cleanupSession();
    }
  };

  // Cleanup session and audio resources
  const cleanupSession = () => {
    stopAIAudioPlayback();

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    isUserSpeakingRef.current = false;

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    setIsCalling(false);
    setIsConnected(false);
    setPipelineState("idle");
    setVolumeLevel(0);
  };

  // Manual interrupt button
  const handleInterrupt = () => {
    stopAIAudioPlayback();
    setPipelineState("listening");
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "interrupt" }));
    }
  };

  // Clear conversation history
  const handleClearHistory = () => {
    setHistory([]);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "clear_history" }));
    }
  };

  // Send manual text message (fallback / testing)
  const handleSendText = (e) => {
    e.preventDefault();
    if (!manualText.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: "text_input", text: manualText.trim() }));
    setManualText("");
  };

  // Determine state indicator colors & labels
  const getStateBadge = () => {
    switch (pipelineState) {
      case "listening":
        return {
          label: "Listening to You...",
          color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
          dotColor: "bg-emerald-400 animate-ping",
        };
      case "processing":
        return {
          label: "Qwen is Thinking...",
          color: "bg-amber-500/20 text-amber-400 border-amber-500/40",
          dotColor: "bg-amber-400 animate-spin-slow",
        };
      case "speaking":
        return {
          label: "AI Speaking (Piper TTS)...",
          color: "bg-purple-500/20 text-purple-300 border-purple-500/40",
          dotColor: "bg-purple-400 animate-pulse",
        };
      default:
        return {
          label: "Idle / Ready to Call",
          color: "bg-slate-700/40 text-slate-400 border-slate-600/40",
          dotColor: "bg-slate-500",
        };
    }
  };

  const stateBadge = getStateBadge();

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", padding: "24px 16px" }}>
      {/* Top Header Bar */}
      <header
        className="glass-panel"
        style={{
          maxWidth: "1000px",
          margin: "0 auto 20px auto",
          width: "100%",
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
              background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 20px rgba(59, 130, 246, 0.4)",
            }}
          >
            <Radio size={22} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: "1.25rem", fontWeight: "700", letterSpacing: "-0.02em" }}>
              AI Calling Agent
            </h1>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Live Voice Conversation • faster-whisper + Qwen + Piper
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {/* User & Assistant Profile Button */}
          <button
            onClick={() => setShowProfileConfig(!showProfileConfig)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              background: "rgba(139, 92, 246, 0.15)",
              border: "1px solid rgba(139, 92, 246, 0.4)",
              color: "#c084fc",
              padding: "8px 14px",
              borderRadius: "10px",
              fontSize: "0.8rem",
              cursor: "pointer",
            }}
            title="Configure User Identity & Assistant Personalization"
          >
            <User size={14} />
            <span>Assistant: {assistantName || "Assistant"}</span>
          </button>

          {/* Supabase Token Button */}
          <button
            onClick={() => setShowTokenInput(!showTokenInput)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              background: token ? "rgba(16, 185, 129, 0.15)" : "rgba(255, 255, 255, 0.05)",
              border: `1px solid ${token ? "rgba(16, 185, 129, 0.4)" : "var(--border-color)"}`,
              color: token ? "#34d399" : "var(--text-muted)",
              padding: "8px 14px",
              borderRadius: "10px",
              fontSize: "0.8rem",
              cursor: "pointer",
            }}
            title="Configure Supabase JWT Token"
          >
            <Key size={14} />
            <span>{token ? "Authenticated" : "Local Dev Auth"}</span>
          </button>

          {/* Connection Status Indicator */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 14px",
              borderRadius: "10px",
              background: isConnected ? "rgba(16, 185, 129, 0.15)" : "rgba(244, 63, 94, 0.15)",
              border: `1px solid ${isConnected ? "rgba(16, 185, 129, 0.3)" : "rgba(244, 63, 94, 0.3)"}`,
              fontSize: "0.8rem",
              color: isConnected ? "#34d399" : "#fb7185",
            }}
          >
            <span
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                backgroundColor: isConnected ? "#34d399" : "#fb7185",
              }}
            />
            {isConnected ? "Connected" : "Disconnected"}
          </div>
        </div>
      </header>

      {/* Role Navigation Bar for Phase 2 Two-Device Calling Simulation */}
      <div
        style={{
          maxWidth: "1000px",
          margin: "0 auto 16px auto",
          width: "100%",
          display: "flex",
          justifyContent: "center",
          gap: "8px",
          background: "rgba(15, 23, 42, 0.6)",
          padding: "6px",
          borderRadius: "14px",
          border: "1px solid var(--border-color)",
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={() => setActiveTab("owner")}
          style={{
            flex: 1,
            minWidth: "160px",
            padding: "10px 16px",
            borderRadius: "10px",
            border: "none",
            background: activeTab === "owner" ? "linear-gradient(135deg, #8b5cf6, #6d28d9)" : "transparent",
            color: activeTab === "owner" ? "#ffffff" : "var(--text-muted)",
            fontWeight: activeTab === "owner" ? "600" : "500",
            fontSize: "0.85rem",
            cursor: "pointer",
            transition: "all 0.15s ease",
            boxShadow: activeTab === "owner" ? "0 2px 10px rgba(139, 92, 246, 0.3)" : "none",
          }}
        >
          💻 Laptop Owner Station
        </button>
        <button
          onClick={() => setActiveTab("caller")}
          style={{
            flex: 1,
            minWidth: "160px",
            padding: "10px 16px",
            borderRadius: "10px",
            border: "none",
            background: activeTab === "caller" ? "linear-gradient(135deg, #10b981, #059669)" : "transparent",
            color: activeTab === "caller" ? "#ffffff" : "var(--text-muted)",
            fontWeight: activeTab === "caller" ? "600" : "500",
            fontSize: "0.85rem",
            cursor: "pointer",
            transition: "all 0.15s ease",
            boxShadow: activeTab === "caller" ? "0 2px 10px rgba(16, 185, 129, 0.3)" : "none",
          }}
        >
          📱 Mobile Caller Screen
        </button>
        <button
          onClick={() => setActiveTab("voice_test")}
          style={{
            flex: 1,
            minWidth: "160px",
            padding: "10px 16px",
            borderRadius: "10px",
            border: "none",
            background: activeTab === "voice_test" ? "linear-gradient(135deg, #3b82f6, #2563eb)" : "transparent",
            color: activeTab === "voice_test" ? "#ffffff" : "var(--text-muted)",
            fontWeight: activeTab === "voice_test" ? "600" : "500",
            fontSize: "0.85rem",
            cursor: "pointer",
            transition: "all 0.15s ease",
            boxShadow: activeTab === "voice_test" ? "0 2px 10px rgba(59, 130, 246, 0.3)" : "none",
          }}
        >
          🧪 Single-Device Voice Test
        </button>
      </div>

      {/* User Identity & Assistant Customization Collapsible */}
      {showProfileConfig && (
        <div
          className="glass-panel"
          style={{
            maxWidth: "1000px",
            margin: "0 auto 20px auto",
            width: "100%",
            padding: "18px 24px",
            display: "flex",
            flexDirection: "column",
            gap: "14px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <User size={18} color="#8b5cf6" />
              <span style={{ fontSize: "0.95rem", fontWeight: "600" }}>User Identity & Assistant Personalization</span>
            </div>
            {profileSavedFeedback && (
              <span style={{ fontSize: "0.8rem", color: "#34d399", display: "flex", alignItems: "center", gap: "4px" }}>
                <Check size={14} /> Profile Saved & Applied!
              </span>
            )}
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Configure who you are and what your AI Calling Assistant should be named. These dynamically populate the Calling Agent system prompt (<code>{"{{user_first_name}}"}</code>, <code>{"{{user_last_name}}"}</code>, <code>{"{{assistant_name}}"}</code>).
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px" }}>
            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "4px" }}>
                User First Name
              </label>
              <input
                type="text"
                placeholder="e.g. Charles or Rudra"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                style={{
                  width: "100%",
                  background: "rgba(0, 0, 0, 0.3)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "8px",
                  padding: "8px 12px",
                  color: "#f8fafc",
                  fontSize: "0.85rem",
                }}
              />
            </div>
            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "4px" }}>
                User Last Name
              </label>
              <input
                type="text"
                placeholder="e.g. Smith or Gandhi"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                style={{
                  width: "100%",
                  background: "rgba(0, 0, 0, 0.3)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "8px",
                  padding: "8px 12px",
                  color: "#f8fafc",
                  fontSize: "0.85rem",
                }}
              />
            </div>
            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "4px" }}>
                Assistant Name
              </label>
              <input
                type="text"
                placeholder="e.g. Alaana or Siri"
                value={assistantName}
                onChange={(e) => setAssistantName(e.target.value)}
                style={{
                  width: "100%",
                  background: "rgba(0, 0, 0, 0.3)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "8px",
                  padding: "8px 12px",
                  color: "#f8fafc",
                  fontSize: "0.85rem",
                }}
              />
            </div>
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              onClick={() => handleSaveProfile(firstName, lastName, assistantName)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                background: "linear-gradient(135deg, #8b5cf6, #7c3aed)",
                color: "#ffffff",
                border: "none",
                borderRadius: "8px",
                padding: "8px 18px",
                fontSize: "0.85rem",
                fontWeight: "500",
                cursor: "pointer",
              }}
            >
              <Check size={14} />
              <span>Save & Apply Profile</span>
            </button>
          </div>
        </div>
      )}

      {/* Supabase Token Input Collapsible */}
      {showTokenInput && (
        <div
          className="glass-panel"
          style={{
            maxWidth: "1000px",
            margin: "0 auto 20px auto",
            width: "100%",
            padding: "16px 20px",
            display: "flex",
            flexDirection: "column",
            gap: "10px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <ShieldCheck size={18} color="#3b82f6" />
            <span style={{ fontSize: "0.9rem", fontWeight: "600" }}>Supabase JWT Bearer Token</span>
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            In development mode, you can leave this blank to connect as a local test user, or paste a Supabase access token to test live multi-tenant authentication.
          </p>
          <div style={{ display: "flex", gap: "8px" }}>
            <input
              type="text"
              placeholder="Paste eyJhbGciOi... access token here"
              value={token}
              onChange={(e) => handleSaveToken(e.target.value)}
              style={{
                flex: 1,
                background: "rgba(0, 0, 0, 0.3)",
                border: "1px solid var(--border-color)",
                borderRadius: "8px",
                padding: "8px 12px",
                color: "#f8fafc",
                fontSize: "0.85rem",
              }}
            />
            {token && (
              <button
                onClick={() => handleSaveToken("")}
                style={{
                  padding: "8px 12px",
                  background: "rgba(244, 63, 94, 0.2)",
                  border: "1px solid rgba(244, 63, 94, 0.4)",
                  color: "#fda4af",
                  borderRadius: "8px",
                  cursor: "pointer",
                }}
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      {/* View Content depending on activeTab */}
      {activeTab === "owner" && (
        <OwnerDashboard
          token={token}
          ownerFirstName={firstName}
          ownerLastName={lastName}
          assistantName={assistantName}
        />
      )}

      {activeTab === "caller" && (
        <CallerInterface
          token={token}
          ownerFirstName={firstName}
          ownerLastName={lastName}
          assistantName={assistantName}
        />
      )}

      {activeTab === "voice_test" && (
        <main
          style={{
            maxWidth: "1000px",
            margin: "0 auto",
            width: "100%",
            flex: 1,
            display: "flex",
            flexDirection: "column",
            gap: "20px",
          }}
        >
        {/* Error Alert */}
        {errorMsg && (
          <div
            style={{
              padding: "12px 16px",
              borderRadius: "12px",
              background: "rgba(244, 63, 94, 0.15)",
              border: "1px solid rgba(244, 63, 94, 0.4)",
              color: "#fecdd3",
              fontSize: "0.85rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <AlertCircle size={18} color="#f43f5e" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg("")}
              style={{ background: "none", border: "none", color: "#fecdd3", cursor: "pointer" }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Live Status & Audio Visualizer Panel */}
        <section
          className="glass-panel"
          style={{
            padding: "24px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            textAlign: "center",
            gap: "20px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* Status Badge */}
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "8px 18px",
              borderRadius: "30px",
              border: "1px solid",
              fontSize: "0.85rem",
              fontWeight: "600",
              letterSpacing: "0.02em",
            }}
            className={stateBadge.color}
          >
            <span
              style={{ width: "8px", height: "8px", borderRadius: "50%" }}
              className={stateBadge.dotColor}
            />
            {stateBadge.label}
          </div>

          {/* Identity Info Banner */}
          <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "6px" }}>
            <span>Assistant: <strong style={{ color: "#c084fc" }}>{assistantName || "Assistant"}</strong></span>
            <span>•</span>
            <span>Representing: <strong style={{ color: "#93c5fd" }}>{firstName || "User"} {lastName || ""}</strong></span>
          </div>

          {/* Voice Wave Visualizer Centerpiece */}
          <div
            style={{
              height: "80px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "6px",
              width: "100%",
              maxWidth: "360px",
            }}
          >
            {isCalling ? (
              Array.from({ length: 24 }).map((_, i) => {
                const heightMultiplier =
                  pipelineState === "speaking"
                    ? 18 + Math.sin(Date.now() / 200 + i) * 14
                    : pipelineState === "listening"
                    ? Math.max(6, volumeLevel * 60 * Math.sin((i / 24) * Math.PI))
                    : 10 + (i % 3) * 6;

                const barColor =
                  pipelineState === "speaking"
                    ? "#a855f7"
                    : pipelineState === "processing"
                    ? "#f59e0b"
                    : volumeLevel > 0.1
                    ? "#10b981"
                    : "rgba(255, 255, 255, 0.2)";

                return (
                  <div
                    key={i}
                    style={{
                      width: "4px",
                      height: `${Math.min(70, Math.max(6, heightMultiplier))}px`,
                      borderRadius: "3px",
                      backgroundColor: barColor,
                      transition: "height 0.08s ease, background-color 0.2s ease",
                    }}
                  />
                );
              })
            ) : (
              <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                Click "Start Conversation" to connect your microphone
              </p>
            )}
          </div>

          {/* Call Controls Button Bar */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexWrap: "wrap",
              gap: "14px",
              width: "100%",
            }}
          >
            {!isCalling ? (
              <button
                onClick={startConversation}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  background: "linear-gradient(135deg, #10b981, #059669)",
                  color: "#ffffff",
                  border: "none",
                  borderRadius: "14px",
                  padding: "14px 28px",
                  fontSize: "1rem",
                  fontWeight: "600",
                  cursor: "pointer",
                  boxShadow: "0 4px 20px rgba(16, 185, 129, 0.4)",
                  transition: "transform 0.15s ease",
                }}
                onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.97)")}
                onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                <PhoneCall size={20} />
                <span>Start Conversation</span>
              </button>
            ) : (
              <>
                {/* End Call Button */}
                <button
                  onClick={cleanupSession}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    background: "linear-gradient(135deg, #e11d48, #be123c)",
                    color: "#ffffff",
                    border: "none",
                    borderRadius: "12px",
                    padding: "12px 22px",
                    fontSize: "0.9rem",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 4px 16px rgba(225, 29, 72, 0.4)",
                  }}
                >
                  <PhoneOff size={18} />
                  <span>End Conversation</span>
                </button>

                {/* Mute Button */}
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    background: isMuted ? "rgba(244, 63, 94, 0.2)" : "rgba(255, 255, 255, 0.08)",
                    border: `1px solid ${isMuted ? "rgba(244, 63, 94, 0.5)" : "var(--border-color)"}`,
                    color: isMuted ? "#fb7185" : "#f8fafc",
                    borderRadius: "12px",
                    padding: "12px 18px",
                    fontSize: "0.9rem",
                    fontWeight: "500",
                    cursor: "pointer",
                  }}
                >
                  {isMuted ? <MicOff size={18} /> : <Mic size={18} />}
                  <span>{isMuted ? "Unmute Mic" : "Mute Mic"}</span>
                </button>

                {/* Interrupt AI Button */}
                <button
                  onClick={handleInterrupt}
                  disabled={pipelineState !== "speaking" && pipelineState !== "processing"}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    background:
                      pipelineState === "speaking" || pipelineState === "processing"
                        ? "linear-gradient(135deg, #8b5cf6, #6d28d9)"
                        : "rgba(255, 255, 255, 0.05)",
                    border: `1px solid ${
                      pipelineState === "speaking" || pipelineState === "processing"
                        ? "rgba(139, 92, 246, 0.6)"
                        : "var(--border-color)"
                    }`,
                    color:
                      pipelineState === "speaking" || pipelineState === "processing"
                        ? "#ffffff"
                        : "var(--text-muted)",
                    borderRadius: "12px",
                    padding: "12px 18px",
                    fontSize: "0.9rem",
                    fontWeight: "500",
                    cursor:
                      pipelineState === "speaking" || pipelineState === "processing"
                        ? "pointer"
                        : "not-allowed",
                    boxShadow:
                      pipelineState === "speaking"
                        ? "0 0 16px rgba(139, 92, 246, 0.5)"
                        : "none",
                  }}
                  title="Interrupt AI speaking immediately (or start speaking into your mic to barge in)"
                >
                  <Square size={16} fill="currentColor" />
                  <span>Interrupt AI</span>
                </button>

                {/* Clear History */}
                <button
                  onClick={handleClearHistory}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    background: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-muted)",
                    borderRadius: "12px",
                    padding: "12px 14px",
                    cursor: "pointer",
                  }}
                  title="Clear conversation history"
                >
                  <Trash2 size={16} />
                </button>
              </>
            )}
          </div>
        </section>

        {/* Conversation Area */}
        <section
          className="glass-panel"
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            minHeight: "380px",
          }}
        >
          {/* Header of Conversation Area */}
          <div
            style={{
              padding: "14px 20px",
              borderBottom: "1px solid var(--border-color)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <Sparkles size={16} color="#3b82f6" />
              <span style={{ fontSize: "0.85rem", fontWeight: "600" }}>Live Conversation Transcript</span>
            </div>
            {sessionId && (
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Session: {sessionId.substring(0, 8)}...
              </span>
            )}
          </div>

          {/* Transcript Message List */}
          <div
            style={{
              flex: 1,
              padding: "20px",
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: "14px",
            }}
          >
            {history.length === 0 ? (
              <div
                style={{
                  margin: "auto",
                  textAlign: "center",
                  color: "var(--text-muted)",
                  padding: "40px 20px",
                }}
              >
                <Mic size={36} style={{ margin: "0 auto 12px auto", opacity: 0.4 }} />
                <p style={{ fontSize: "0.95rem", fontWeight: "500", marginBottom: "4px" }}>
                  No messages yet
                </p>
                <p style={{ fontSize: "0.8rem", maxWidth: "340px" }}>
                  Start the conversation and speak naturally. Faster-whisper will transcribe your speech and Qwen will respond through Piper TTS.
                </p>
              </div>
            ) : (
              history.map((msg, idx) => {
                const isUser = msg.role === "user";
                return (
                  <div
                    key={idx}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: isUser ? "flex-end" : "flex-start",
                      maxWidth: "80%",
                      alignSelf: isUser ? "flex-end" : "flex-start",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--text-muted)",
                        marginBottom: "4px",
                        marginLeft: isUser ? "0" : "8px",
                        marginRight: isUser ? "8px" : "0",
                      }}
                    >
                      {isUser ? `${firstName || "You"} (Microphone)` : `${assistantName || "Assistant"} (Personal Receptionist)`}
                    </span>
                    <div
                      style={{
                        padding: "12px 16px",
                        borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                        background: isUser
                          ? "linear-gradient(135deg, #2563eb, #1d4ed8)"
                          : "rgba(30, 41, 59, 0.8)",
                        border: isUser
                          ? "1px solid rgba(59, 130, 246, 0.5)"
                          : "1px solid rgba(255, 255, 255, 0.08)",
                        color: "#f8fafc",
                        fontSize: "0.95rem",
                        lineHeight: "1.45",
                        boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
                      }}
                    >
                      {msg.content}
                    </div>
                  </div>
                );
              })
            )}

            {/* In-flight typing / thinking indicator */}
            {pipelineState === "processing" && (
              <div
                style={{
                  alignSelf: "flex-start",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "10px 16px",
                  borderRadius: "16px 16px 16px 4px",
                  background: "rgba(30, 41, 59, 0.6)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  color: "#f59e0b",
                  fontSize: "0.85rem",
                }}
              >
                <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#f59e0b" }} className="animate-ping" />
                <span>Qwen is preparing spoken response...</span>
              </div>
            )}

            <div ref={chatBottomRef} />
          </div>

          {/* Bottom Manual Text Input (Fallback / Testing) */}
          <form
            onSubmit={handleSendText}
            style={{
              padding: "12px 16px",
              borderTop: "1px solid var(--border-color)",
              background: "rgba(10, 14, 23, 0.6)",
              display: "flex",
              gap: "8px",
            }}
          >
            <input
              type="text"
              placeholder={isCalling ? "Or type a message to test voice response..." : "Start conversation to enable speech and text..."}
              disabled={!isCalling}
              value={manualText}
              onChange={(e) => setManualText(e.target.value)}
              style={{
                flex: 1,
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid var(--border-color)",
                borderRadius: "10px",
                padding: "10px 14px",
                color: "#f8fafc",
                fontSize: "0.9rem",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={!isCalling || !manualText.trim()}
              style={{
                padding: "10px 16px",
                borderRadius: "10px",
                background: isCalling && manualText.trim() ? "#3b82f6" : "rgba(255, 255, 255, 0.05)",
                border: "none",
                color: "#ffffff",
                cursor: isCalling && manualText.trim() ? "pointer" : "not-allowed",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Send size={16} />
            </button>
          </form>
        </section>
      </main>
      )}
    </div>
  );
}
