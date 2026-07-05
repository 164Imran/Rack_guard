"use client";

import {
  type CSSProperties,
  type WheelEvent,
  useEffect,
  useRef,
  useState
} from "react";
import {
  ArrowLeft,
  ArrowRight,
  Bot,
  Check,
  CheckCircle2,
  CircleAlert,
  Cpu,
  Lightbulb,
  LoaderCircle,
  Mic,
  Radio,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  Speaker,
  TriangleAlert
} from "lucide-react";
import {
  rackFleet,
  type GpuSnapshot,
  type RackSnapshot
} from "@/lib/mockRackFleet";
import { loadGpuDecision } from "@/lib/rackGuardianApi";
import type { GpuDecisionData } from "@/lib/rackGuardianTypes";

type RackState = "safe" | "warning" | "critical";
type SpeechState = "idle" | "loading" | "playing" | "error";
type MicState = "idle" | "recording" | "transcribing" | "error";
type CopilotSource = "crusoe" | "fallback";
type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  source?: CopilotSource;
};

const VOICE_API_URL = process.env.NEXT_PUBLIC_VOICE_API_URL || "http://127.0.0.1:8000";

type Rack = RackSnapshot & {
  id: string;
  temperature: number;
  predicted: number;
  load: number;
  safeLimit: number;
  slowdown: string;
  state: RackState;
  cause: string;
  secondaryCause: string;
  recommendation: string;
};

const racks: Rack[] = rackFleet.map((rack) => {
  const priorityGpu = [...rack.gpus].sort(
    (a, b) => b.predicted_temperature_c - a.predicted_temperature_c
  )[0];
  return {
    ...rack,
    id: rack.rack_id,
    temperature: priorityGpu.temperature_c,
    predicted: priorityGpu.predicted_temperature_c,
    load: priorityGpu.utilization_percent,
    safeLimit: 85,
    state: rack.status === "normal" ? "safe" : rack.status,
    cause: rack.cause,
    secondaryCause: rack.secondary_cause,
    recommendation: rack.recommendation
  };
});

const palette: Record<RackState, CSSProperties> = {
  safe: { "--accent": "#48d7ef", "--glow": "rgba(72,215,239,.22)", "--fill": "rgba(30,105,121,.22)" } as CSSProperties,
  warning: { "--accent": "#f0a84b", "--glow": "rgba(240,168,75,.28)", "--fill": "rgba(119,72,24,.24)" } as CSSProperties,
  critical: { "--accent": "#f05288", "--glow": "rgba(240,82,136,.42)", "--fill": "rgba(119,26,67,.28)" } as CSSProperties
};

export default function DataCenterRoom() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [selectedRackIndex, setSelectedRackIndex] = useState<number | null>(null);
  const [selectedGpuId, setSelectedGpuId] = useState<string | null>(null);
  const [accepted, setAccepted] = useState(false);
  const [question, setQuestion] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [copilotLoading, setCopilotLoading] = useState(false);
  const [speechState, setSpeechState] = useState<SpeechState>("idle");
  const [micState, setMicState] = useState<MicState>("idle");
  const wheelLocked = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micContextRef = useRef<AudioContext | null>(null);
  const micProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const micChunksRef = useRef<Float32Array[]>([]);
  const micTimerRef = useRef<number | null>(null);

  const rack = racks[activeIndex];
  const outcome = rack.id === "R-04" ? 74 : Math.max(65, rack.predicted - 7);
  const riskRackCount = racks.filter((item) => item.status !== "normal").length;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  useEffect(() => () => {
    audioRef.current?.pause();
    micStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (micTimerRef.current) window.clearTimeout(micTimerRef.current);
  }, []);

  function move(direction: number) {
    setActiveIndex((current) => Math.max(0, Math.min(racks.length - 1, current + direction)));
    setSelectedRackIndex(null);
    setSelectedGpuId(null);
    setAccepted(false);
  }

  function select(index: number) {
    setActiveIndex(index);
    setSelectedRackIndex(index);
    setSelectedGpuId(null);
    setAccepted(false);
  }

  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    if (wheelLocked.current || Math.abs(event.deltaY) < 8) return;
    wheelLocked.current = true;
    move(event.deltaY > 0 ? 1 : -1);
    window.setTimeout(() => { wheelLocked.current = false; }, 280);
  }

  async function speakAnalysis() {
    if (speechState === "loading") return;
    if (speechState === "playing") {
      audioRef.current?.pause();
      if (audioRef.current?.src.startsWith("blob:")) URL.revokeObjectURL(audioRef.current.src);
      audioRef.current = null;
      setSpeechState("idle");
      return;
    }
    const text = accepted
      ? `Action accepted. ${rack.id} is now converging toward ${outcome} degrees.`
      : `${rack.id} is predicted to reach ${rack.predicted} degrees. ${rack.recommendation}`;

    audioRef.current?.pause();
    if (audioRef.current?.src.startsWith("blob:")) URL.revokeObjectURL(audioRef.current.src);
    setSpeechState("loading");

    try {
      const response = await fetch(`${VOICE_API_URL}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      if (!response.ok) throw new Error("TTS unavailable");

      const audioUrl = URL.createObjectURL(await response.blob());
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      audio.onplay = () => setSpeechState("playing");
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setSpeechState("idle");
      };
      audio.onerror = () => {
        URL.revokeObjectURL(audioUrl);
        setSpeechState("error");
      };
      await audio.play();
    } catch {
      setSpeechState("error");
      window.setTimeout(() => setSpeechState("idle"), 2400);
    }
  }

  async function toggleMicrophone() {
    if (micState === "recording") {
      await stopRecording();
      return;
    }
    if (micState === "transcribing") return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      micChunksRef.current = [];
      processor.onaudioprocess = (event) => {
        micChunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(context.destination);
      micStreamRef.current = stream;
      micContextRef.current = context;
      micProcessorRef.current = processor;
      setMicState("recording");
      micTimerRef.current = window.setTimeout(() => void stopRecording(), 8000);
    } catch {
      setMicState("error");
      window.setTimeout(() => setMicState("idle"), 2400);
    }
  }

  async function stopRecording() {
    if (micTimerRef.current) window.clearTimeout(micTimerRef.current);
    micProcessorRef.current?.disconnect();
    micStreamRef.current?.getTracks().forEach((track) => track.stop());
    const sampleRate = micContextRef.current?.sampleRate || 48000;
    await micContextRef.current?.close();
    micProcessorRef.current = null;
    micStreamRef.current = null;
    micContextRef.current = null;

    const audio = mergeAudioChunks(micChunksRef.current);
    if (audio.length < sampleRate / 4) {
      setMicState("idle");
      return;
    }

    setMicState("transcribing");
    try {
      const response = await fetch(`${VOICE_API_URL}/stt`, {
        method: "POST",
        headers: { "Content-Type": "audio/wav" },
        body: encodeWav(audio, sampleRate)
      });
      if (!response.ok) throw new Error("STT unavailable");
      const result = await response.json();
      setQuestion(typeof result.text === "string" ? result.text : "");
      setMicState("idle");
    } catch {
      setMicState("error");
      window.setTimeout(() => setMicState("idle"), 2400);
    }
  }

  async function askGuardian() {
    const message = question.trim();
    if (!message || copilotLoading) return;

    const selectedRack = selectedRackIndex === null ? null : racks[selectedRackIndex];
    const selectedGpu = selectedRack?.gpus.find((gpu) => gpu.gpu_id === selectedGpuId) ?? null;
    const highestRiskRack = racks[0];
    const highestRiskGpu = [...highestRiskRack.gpus].sort(
      (a, b) => b.risk_score - a.risk_score,
    )[0];
    const fleetSummary = {
      rack_count: racks.length,
      gpu_count: racks.reduce((total, item) => total + item.gpu_count, 0),
      racks_to_check: racks.filter((item) => item.gpus_to_check > 0).length,
      gpus_to_check: racks.reduce((total, item) => total + item.gpus_to_check, 0),
      highest_risk_rack: highestRiskRack.rack_id,
      highest_risk_gpu: highestRiskGpu.gpu_id,
    };

    setChatMessages((current) => [
      ...current,
      { id: Date.now(), role: "user", content: message } as ChatMessage,
    ].slice(-6));
    setQuestion("");
    setCopilotLoading(true);

    try {
      const response = await fetch("/api/copilot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          context: { selectedRack, selectedGpu, fleetSummary },
        }),
      });
      const result = await response.json() as {
        ok?: boolean;
        answer?: string;
        source?: CopilotSource;
        error?: string;
      };

      if (!response.ok || !result.ok || !result.answer) {
        throw new Error(result.error || "Copilot response unavailable.");
      }

      setChatMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: result.answer!,
          source: result.source ?? "fallback",
        } as ChatMessage,
      ].slice(-6));
    } catch {
      setChatMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: "The copilot is temporarily unavailable. Your rack and GPU telemetry remain visible.",
          source: "fallback",
        } as ChatMessage,
      ].slice(-6));
    } finally {
      setCopilotLoading(false);
    }
  }

  return (
    <section className="rg-shell">
      <AmbientCanvas state={accepted ? "safe" : rack.state} />

      <div className="rg-main">
        <header className="rg-header">
          <div>
            <div className="rg-kicker"><Radio /> Live thermal co-pilot</div>
            <h1>Rack Guardian</h1>
          </div>
          <div className="rg-alert"><CircleAlert /><div><strong>{riskRackCount} racks need review</strong><span>{racks[0].id} has the highest priority</span></div></div>
        </header>

        <div className="layer-stage">
          <OverviewLayer
            activeIndex={activeIndex}
            selectedIndex={selectedRackIndex}
            selectedGpuId={selectedGpuId}
            accepted={accepted}
            onMove={move}
            onSelect={select}
            onSelectGpu={setSelectedGpuId}
            onBackToRack={() => setSelectedGpuId(null)}
            onBackToRacks={() => {
              setSelectedRackIndex(null);
              setSelectedGpuId(null);
            }}
            onWheel={handleWheel}
          />
        </div>

        <div className="rg-statusbar" style={palette[rack.state]}>
          <Stat icon={Cpu} label="GPUs" value={`${rack.gpu_count}`} />
          <Stat icon={CircleAlert} label="To check" value={`${rack.gpus_to_check}`} accent />
          <Stat icon={TriangleAlert} label="Warning" value={`${rack.warning_count}`} />
          <Stat icon={ShieldCheck} label="Critical" value={`${rack.critical_count}`} />
        </div>
      </div>

      <aside className="rg-ai">
        <header className="ai-head">
          <div className={`ai-orb ${accepted ? "orb-safe" : ""}`}><span /></div>
          <div><div className="ai-name"><h2>Guardian AI</h2><span>Live</span></div><p>Decision co-pilot</p></div>
          <button
            className={`speech-button speech-${speechState}`}
            type="button"
            aria-label={speechState === "loading" ? "Generating speech" : speechState === "playing" ? "Stop speech" : "Speak analysis with Gradium"}
            title={speechState === "error" ? "Gradium is unavailable. Check the server API key." : "Speak with Gradium"}
            onClick={speakAnalysis}
            disabled={speechState === "loading"}
          >
            {speechState === "loading" ? <LoaderCircle /> : <Speaker />}
          </button>
        </header>

        <div className="ai-sync"><Cpu /><span>Context synced with {rack.id}</span><b>#{activeIndex + 1}</b></div>

        <div className="ai-body" aria-live="polite">
          <div className="ai-bubble">
            <div className="bot-mark"><Bot /></div>
            <div>
              <span>Assessment</span>
              <p>
                {`${rack.id} contains ${rack.gpu_count} GPUs. ${rack.gpus_to_check} require review: ${rack.warning_count} warning and ${rack.critical_count} critical.`}
              </p>
            </div>
          </div>

          <div className="ai-cause">
            <span><Sparkles /> Why</span>
            <strong>{rack.cause}</strong>
            <p>{rack.secondaryCause}.</p>
          </div>

          <div className={`ai-action-card ${accepted ? "action-safe" : ""}`}>
            <span>{accepted ? <Check /> : <Lightbulb />} {accepted ? "Outcome" : "Recommended action"}</span>
            <strong>{accepted ? `${rack.predicted}°C → ${outcome}°C` : rack.recommendation}</strong>
            <p>{rack.state === "critical" ? "Thermal slowdown avoided." : "The rack remains inside its safe envelope."}</p>
          </div>
          {chatMessages.length > 0 && (
            <div className="ai-conversation" aria-label="Copilot conversation">
              {chatMessages.map((message) => (
                <div key={message.id} className={`ai-message message-${message.role}`}>
                  <span>
                    {message.role === "user" ? "You" : "Guardian"}
                    {message.source && <b>{message.source === "crusoe" ? "Crusoe" : "Fallback"}</b>}
                  </span>
                  <p>{message.content}</p>
                </div>
              ))}
              {copilotLoading && <div className="ai-thinking"><LoaderCircle />Thinking…</div>}
            </div>
          )}
        </div>

        <div className="ai-buttons">
          <button className="accept" type="button" onClick={() => setAccepted(true)} disabled={accepted || rack.state === "safe"}>{accepted ? <Check /> : <ShieldCheck />}{accepted ? "Accepted" : "Accept recommendation"}</button>
        </div>
        <div className="ai-input">
          <input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void askGuardian(); }} placeholder={`Ask about ${selectedGpuId ?? rack.id}…`} aria-label="Ask Guardian AI" disabled={copilotLoading} />
          <button
            className={`mic-button mic-${micState}`}
            type="button"
            aria-label={micState === "recording" ? "Stop recording" : micState === "transcribing" ? "Transcribing speech" : "Start voice command"}
            title={micState === "error" ? "Microphone or Gradium STT unavailable" : "Voice command with Gradium"}
            onClick={toggleMicrophone}
            disabled={micState === "transcribing"}
          >
            {micState === "transcribing" ? <LoaderCircle /> : <Mic />}
          </button>
          <button className={copilotLoading ? "send-loading" : ""} type="button" aria-label="Send message" onClick={() => void askGuardian()} disabled={!question.trim() || copilotLoading}>{copilotLoading ? <LoaderCircle /> : <Send />}</button>
        </div>
        <p className="ai-note">Recommendation only. You remain in control.</p>
      </aside>

      <style jsx global>{`
        html, body { height: 100%; overflow: hidden; }
        .rg-shell {
          position: relative; isolation: isolate; display: grid; grid-template-columns: minmax(0,1fr) 370px;
          width: 100%; height: calc(100vh - 24px); overflow: hidden; border: 1px solid rgba(255,255,255,.09);
          border-radius: 10px; background: #080c12; color: #eff7fb; font-family: Inter,ui-sans-serif,system-ui,sans-serif;
          box-shadow: 0 30px 90px rgba(0,0,0,.42);
        }
        .ambient-canvas { position: absolute; inset: 0; z-index: -1; width: 100%; height: 100%; opacity: .68; }
        .rg-shell.roi-mode{grid-template-columns:minmax(0,1fr)}.roi-mode .rg-ai{display:none}
        .rg-main { display: grid; grid-template-rows: auto minmax(0,1fr) auto; gap:12px; min-width: 0; min-height: 0; overflow: hidden; padding: 18px 22px; }
        .rg-header { display: flex; align-items: center; justify-content: space-between; gap: 20px; border-bottom: 1px solid rgba(255,255,255,.08); padding-bottom: 13px; }
        .rg-kicker { display: flex; align-items: center; gap: 7px; color: #66ddf3; font-size: 9px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .rg-kicker svg { width: 12px; height: 12px; }
        .rg-header h1 { margin: 4px 0 0; font-size: 25px; }
        .rg-alert { display: flex; align-items: center; gap: 9px; border: 1px solid rgba(240,82,136,.25); border-radius: 7px; background: rgba(240,82,136,.07); padding: 8px 11px; color: #ff78a5; }
        .rg-alert svg { width: 16px; height: 16px; }.rg-alert strong,.rg-alert span { display:block }.rg-alert strong{font-size:10px}.rg-alert span{margin-top:2px;color:#9a7182;font-size:8px}
        .layer-tabs { display: flex; align-items: center; gap: 6px; padding: 12px 0; }
        .layer-button { display: inline-flex; min-height: 38px; align-items: center; gap: 7px; border: 1px solid transparent; border-radius: 6px; background: transparent; padding: 0 12px; color: #708092; font-size: 11px; font-weight: 650; }
        .layer-button svg { width: 14px; height: 14px; }.layer-button:hover{background:rgba(255,255,255,.035);color:#cbd6df}.layer-button.active{border-color:rgba(72,215,239,.2);background:rgba(72,215,239,.075);color:#83e4f5}
        .layer-button kbd{margin-left:3px;border:1px solid rgba(255,255,255,.09);border-radius:3px;padding:1px 4px;color:#526273;font-size:8px}
        .layer-context{display:flex;align-items:center;gap:6px;margin-left:auto;color:#627385;font-size:9px;text-transform:uppercase}.layer-context strong{color:#dce7ee;font-size:11px}
        .layer-stage { min-height: 0; overflow: hidden; border: 1px solid rgba(255,255,255,.075); border-radius: 8px; background: rgba(9,15,22,.55); }
        .overview-layer { position: relative; width: 100%; height: 100%; min-height: 390px; overflow: hidden; user-select:none; }
        .deck-caption { position:absolute;top:16px;left:18px;z-index:5 }.deck-caption span,.deck-caption strong{display:block}.deck-caption span{color:#657587;font-size:9px;letter-spacing:.1em;text-transform:uppercase}.deck-caption strong{margin-top:3px;font-size:13px}
        .deck-controls{position:absolute;top:14px;right:16px;z-index:8;display:flex;gap:6px}.deck-controls button{display:grid;width:38px;height:38px;place-items:center;border:1px solid rgba(255,255,255,.09);border-radius:6px;background:rgba(255,255,255,.03);color:#8292a2}.deck-controls button:hover:not(:disabled){background:rgba(255,255,255,.07);color:#fff}.deck-controls button:disabled{cursor:not-allowed;opacity:.28}.deck-controls svg{width:15px}
        .rack-deck { position:absolute;inset:56px 0 0;perspective:1000px; }
        .deck-rack { position:absolute;left:50%;top:50%;width:180px;height:310px;opacity:var(--deck-opacity);transform:translate(-50%,-50%) translateX(calc(var(--deck-offset) * 196px)) translateZ(calc((1 - var(--deck-distance)) * 55px)) rotateY(calc(var(--deck-offset) * -4deg)) scale(calc(1 - var(--deck-distance) * .09));z-index:var(--deck-z);transition:transform .32s cubic-bezier(.22,.8,.25,1),opacity .25s;pointer-events:var(--deck-events)}
        .tower { position:relative;display:flex;width:100%;height:100%;flex-direction:column;border:1px solid color-mix(in srgb,var(--accent) 38%,transparent);border-radius:5px 5px 8px 8px;background:repeating-linear-gradient(0deg,transparent 0 15px,rgba(255,255,255,.065) 15px 16px),linear-gradient(145deg,var(--fill),rgba(10,16,24,.98) 48%);padding:16px 14px;color:#edf6fa;text-align:left;box-shadow:inset 0 0 34px rgba(0,0,0,.4),0 18px 35px rgba(0,0,0,.3)}
        .tower::before{content:"";position:absolute;top:-8px;right:5px;left:5px;height:8px;border:1px solid color-mix(in srgb,var(--accent) 28%,transparent);border-bottom:0;background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 15%,#111a25),#090f16);clip-path:polygon(6% 100%,0 40%,10% 0,92% 0,100% 50%,94% 100%)}
        .tower[aria-pressed=true]{box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 25%,transparent),0 0 36px var(--glow),0 22px 38px rgba(0,0,0,.4)}.tower-critical[aria-pressed=true]{animation:thermal-pulse 2.8s ease-in-out infinite}
        .tower-rank{color:#6b7b8c;font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.tower-head{display:flex;align-items:center;gap:7px;margin-top:12px}.tower-head svg{width:14px}.tower-head strong{font-size:15px}.tower-head i{margin-left:auto;color:var(--accent)}.tower-temp{margin-top:22px;color:var(--accent);font-size:42px;font-weight:750;line-height:1;text-shadow:0 0 18px var(--glow)}.tower-temp small{font-size:14px;color:#8796a6}.tower-slots{display:grid;gap:6px;margin-top:auto;padding-top:20px}.tower-slots span{position:relative;height:6px;border-radius:1px;background:rgba(255,255,255,.07)}.tower-slots span::before{content:"";position:absolute;top:2px;left:4px;width:2px;height:2px;border-radius:50%;background:var(--accent);box-shadow:0 0 5px var(--accent)}.tower-load{display:flex;align-items:center;gap:8px;margin-top:13px}.tower-track{flex:1;height:4px;border-radius:4px;background:rgba(255,255,255,.08);overflow:hidden}.tower-track span{display:block;height:100%;background:var(--accent)}.tower-load b{color:#7c8b9b;font-size:9px}.tower-foot{display:flex;justify-content:space-between;margin-top:12px;border-top:1px solid rgba(255,255,255,.07);padding-top:9px;color:#68798a;font-size:9px}.tower-foot strong{color:var(--accent);font-size:11px}
        .tower-gpu-count{display:flex;align-items:baseline;gap:7px;margin-top:18px;color:var(--accent)}.tower-gpu-count strong{font-size:36px;line-height:1}.tower-gpu-count span{color:#8292a2;font-size:10px;font-weight:700;text-transform:uppercase}
        .tower-risk-counts{display:grid;grid-template-columns:1fr;gap:3px;margin-top:10px}.tower-risk-counts span{color:#748494;font-size:9px}.tower-risk-counts b{display:inline-block;width:14px;color:#d8e3e9}
        .with-gpu-grid .rack-deck{right:64%}.with-gpu-grid .deck-rack{top:14px;width:128px;height:248px;transform:translate(-50%,0) translateX(calc(var(--deck-offset) * 118px)) translateZ(calc((1 - var(--deck-distance)) * 35px)) rotateY(calc(var(--deck-offset) * -4deg)) scale(calc(1 - var(--deck-distance) * .11))}
        .with-gpu-grid .tower{padding:13px 11px}.with-gpu-grid .tower-head{margin-top:8px}.with-gpu-grid .tower-gpu-count{margin-top:12px}.with-gpu-grid .tower-gpu-count strong{font-size:29px}.with-gpu-grid .tower-risk-counts{margin-top:7px}.with-gpu-grid .tower-slots{gap:4px;padding-top:9px}.with-gpu-grid .tower-load{margin-top:8px}.with-gpu-grid .tower-foot{margin-top:7px;padding-top:6px}
        .gpu-rack-panel{position:absolute;top:56px;right:0;bottom:0;width:62%;z-index:9;border-left:1px solid rgba(255,255,255,.08);background:rgba(8,14,21,.94);padding:14px 16px 16px;animation:gpu-panel-in .22s ease-out}.gpu-rack-panel>header{display:flex;align-items:center;justify-content:space-between;gap:12px;height:42px}.gpu-rack-panel>header span{color:#667789;font-size:8px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.gpu-rack-panel>header h2{margin:3px 0 0;font-size:15px}.gpu-panel-status{border:1px solid rgba(255,255,255,.1);border-radius:99px;padding:5px 8px;color:#b6c2cc;font-size:9px;font-weight:700}.gpu-panel-status.status-critical{border-color:rgba(240,82,136,.28);color:#f4779f}.gpu-panel-status.status-warning{border-color:rgba(240,168,75,.28);color:#f0b86e}.gpu-panel-status.status-normal{border-color:rgba(69,197,138,.28);color:#69dca2}
        .gpu-card-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(4,minmax(0,1fr));gap:7px;height:calc(100% - 48px);margin-top:6px}.gpu-nav-card{min-width:0;border:1px solid rgba(255,255,255,.075);border-radius:6px;background:rgba(15,24,34,.78);padding:9px 10px;color:inherit;text-align:left;cursor:pointer;transition:border-color .18s ease,background .18s ease,transform .18s ease}.gpu-nav-card:hover{border-color:rgba(72,215,239,.35);background:rgba(21,34,47,.96);transform:translateY(-1px)}.gpu-nav-card:focus-visible{outline:2px solid #48d7ef;outline-offset:2px}.gpu-nav-card.gpu-critical{border-color:rgba(240,82,136,.25);background:rgba(240,82,136,.045)}.gpu-nav-card.gpu-warning{border-color:rgba(240,168,75,.22);background:rgba(240,168,75,.035)}.gpu-nav-card.gpu-normal{border-color:rgba(72,215,239,.11)}
        .gpu-nav-head{display:flex;align-items:center;justify-content:space-between;gap:7px}.gpu-nav-head>span,.gpu-nav-head>i{display:flex;align-items:center;gap:5px}.gpu-nav-head>span{color:#dce7ed;font-size:10px;font-weight:700}.gpu-nav-head>span svg{width:11px;color:#69d9ed}.gpu-nav-head>i{color:#718293;font-size:7px;font-style:normal;text-transform:uppercase}.gpu-nav-head>i svg{width:9px}.gpu-critical .gpu-nav-head>i,.gpu-critical .gpu-nav-temp{color:#f4779f}.gpu-warning .gpu-nav-head>i,.gpu-warning .gpu-nav-temp{color:#f0b86e}.gpu-normal .gpu-nav-head>i,.gpu-normal .gpu-nav-temp{color:#69dca2}
        .gpu-nav-temp{margin-top:5px;font-size:22px;font-weight:750;line-height:1}.gpu-nav-temp small{font-size:9px}.gpu-nav-card dl{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin:7px 0 0;border-top:1px solid rgba(255,255,255,.06);padding-top:6px}.gpu-nav-card dt{overflow:hidden;color:#586a7b;font-size:7px;text-overflow:ellipsis;white-space:nowrap}.gpu-nav-card dd{margin:2px 0 0;color:#b7c4cd;font-size:8px;font-weight:700}
        .gpu-detail{display:flex;flex-direction:column;gap:9px;overflow:hidden}.gpu-detail>header{height:34px;flex:0 0 auto}.gpu-back{display:inline-flex;align-items:center;gap:6px;min-height:32px;border:0;background:transparent;color:#8fa1b2;font:inherit;font-size:10px;font-weight:700;cursor:pointer}.gpu-back svg{width:14px}.gpu-back:hover{color:#f4f8fb}.gpu-back:focus-visible{outline:2px solid #48d7ef;outline-offset:2px}.gpu-detail-status{display:flex;align-items:center;gap:7px}.gpu-loading,.gpu-source{display:inline-flex;align-items:center;gap:4px;color:#6f8191;font-size:8px;font-weight:700;text-transform:uppercase}.gpu-loading svg{width:11px;animation:speech-spin .8s linear infinite}.gpu-detail-title{display:flex;align-items:flex-end;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.07);padding-bottom:8px}.gpu-detail-title span{color:#48d7ef;font-size:9px;font-weight:750;text-transform:uppercase}.gpu-detail-title h2{margin:0;font-size:16px}.gpu-snapshot{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;overflow:hidden;border:1px solid rgba(255,255,255,.08);border-radius:7px;background:rgba(255,255,255,.06)}.gpu-snapshot>span{display:flex;min-width:0;flex-direction:column;gap:3px;background:#0d151e;padding:10px}.gpu-snapshot small,.gpu-decision-block small,.gpu-sim-flow small,.gpu-details small{color:#647587;font-size:8px;font-weight:700;text-transform:uppercase}.gpu-snapshot strong{font-size:16px}.gpu-decision-block{border:1px solid rgba(72,215,239,.13);border-radius:7px;background:rgba(13,21,30,.82);padding:11px 12px}.gpu-decision-block.simulation{border-color:rgba(255,255,255,.09)}.gpu-block-heading{display:flex;align-items:center;gap:9px}.gpu-block-heading>svg{width:17px;color:#48d7ef}.gpu-block-heading span,.gpu-recommendation span{display:flex;flex-direction:column;gap:2px}.gpu-block-heading strong,.gpu-recommendation strong{font-size:12px}.gpu-decision-block p{margin:8px 0 0;color:#9ba9b7;font-size:10px;line-height:1.5}.gpu-decision-block p b{color:#c4d0d8}.gpu-recommendation{display:flex;align-items:center;gap:8px;margin-top:9px;border-top:1px solid rgba(255,255,255,.06);padding-top:8px}.gpu-recommendation>svg{width:15px;color:#f0a84b}.gpu-details{min-height:0;overflow:auto;border-top:1px solid rgba(255,255,255,.07);padding-top:2px}.gpu-details summary{display:flex;min-height:32px;align-items:center;color:#8fa1b2;font-size:9px;font-weight:700;cursor:pointer;list-style:none}.gpu-details summary::-webkit-details-marker{display:none}.gpu-details summary::after{content:"+";margin-left:auto;color:#617384;font-size:14px}.gpu-details[open] summary::after{content:"−"}.gpu-details-content{display:grid;gap:7px;padding:2px 0 6px}.gpu-details-content section{border-left:1px solid rgba(255,255,255,.08);padding-left:9px}.gpu-details-content p{margin:3px 0 0;color:#96a5b2;font-size:9px;line-height:1.45}.gpu-details-content ol{display:grid;gap:5px;margin:5px 0 0;padding:0;list-style:none}.gpu-details-content li{display:flex;justify-content:space-between;gap:8px;color:#a9b6c0;font-size:9px}.gpu-details-content li>span{display:grid;gap:2px}.gpu-details-content li>span small{font-size:7px;font-weight:500;line-height:1.35;text-transform:none}.gpu-details-content li b{color:#69d9ed}.gpu-sim-flow{display:grid;grid-template-columns:1fr 24px 1fr;align-items:center;gap:7px;margin-top:9px}.gpu-sim-flow>svg{width:15px;color:#526475}.gpu-sim-flow>span{display:grid;grid-template-columns:auto 1fr;align-items:center;gap:3px 8px;border-radius:5px;background:rgba(255,255,255,.035);padding:7px 8px}.gpu-sim-flow strong{font-size:17px}.gpu-sim-flow i{grid-column:1/-1;color:#8393a2;font-size:8px;font-style:normal;text-transform:capitalize}
        .analysis-layer,.simulation-layer{display:grid;width:100%;height:100%;min-height:390px;padding:22px}.analysis-layer{grid-template-columns:1.1fr .9fr;gap:16px}.analysis-summary,.cause-panel,.simulation-before,.simulation-after{border:1px solid rgba(255,255,255,.08);border-radius:7px;background:rgba(14,22,32,.72);padding:18px}.analysis-summary h2,.simulation-layer h2{margin:5px 0 0;font-size:22px}.eyebrow{color:#68798b;font-size:9px;font-weight:700;letter-spacing:.11em;text-transform:uppercase}.analysis-summary>p{color:#aab7c4;font-size:13px;line-height:1.6}.trajectory{position:relative;height:140px;margin-top:20px}.trajectory-line{position:absolute;right:0;left:0;height:2px;background:rgba(255,255,255,.07)}.trajectory-line.safe{top:54px;border-top:1px dashed #56697a;background:transparent}.trajectory-line.safe::before{content:"85°C safe limit";position:absolute;top:-18px;color:#68798a;font-size:9px}.trajectory svg{width:100%;height:100%}.cause-panel{display:flex;flex-direction:column;justify-content:center}.cause-row{margin-top:18px}.cause-row div:first-child{display:flex;justify-content:space-between;color:#aab7c4;font-size:11px}.cause-bar{height:5px;margin-top:7px;border-radius:5px;background:rgba(255,255,255,.07);overflow:hidden}.cause-bar span{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#28bad6,#f05288)}
        .simulation-layer{grid-template-columns:1fr auto 1fr;align-items:center;gap:18px}.simulation-arrow{color:#55d7ef}.simulation-arrow svg{width:24px}.sim-temp{margin-top:22px;font-size:52px;font-weight:750}.simulation-before .sim-temp{color:#f4779f}.simulation-after .sim-temp{color:#69dca2}.sim-state{display:inline-flex;margin-top:12px;border:1px solid rgba(255,255,255,.09);border-radius:99px;padding:5px 9px;color:#9eacba;font-size:10px}.simulation-after{border-color:rgba(69,197,138,.2);background:rgba(69,197,138,.04)}.run-simulation{display:inline-flex;min-height:42px;align-items:center;gap:7px;margin-top:22px;border:1px solid #20bedb;border-radius:6px;background:#1198b2;padding:0 16px;color:#041419;font-size:11px;font-weight:700}.run-simulation svg{width:14px}
        .rg-statusbar{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:10px;border:1px solid rgba(255,255,255,.08);border-radius:7px;background:rgba(12,19,28,.76);padding:11px 14px}.rg-stat{display:flex;align-items:center;gap:8px}.rg-stat svg{width:14px;color:#627385}.rg-stat span,.rg-stat strong{display:block}.rg-stat span{color:#68798a;font-size:8px;text-transform:uppercase}.rg-stat strong{margin-top:2px;color:#dce7ee;font-size:12px}.rg-stat.accent strong{color:var(--accent)}
        .rg-ai{display:flex;min-width:0;flex-direction:column;border-left:1px solid rgba(255,255,255,.09);background:rgba(8,13,20,.96);padding:18px}.ai-head{display:flex;align-items:center;gap:10px;border-bottom:1px solid rgba(255,255,255,.08);padding-bottom:14px}.ai-orb{position:relative;display:grid;width:42px;height:42px;place-items:center;border:1px solid rgba(72,215,239,.25);border-radius:50%}.ai-orb::before{content:"";position:absolute;inset:7px;border:1px solid rgba(72,215,239,.3);border-radius:50%;animation:orb 3s ease-in-out infinite}.ai-orb span{width:13px;height:13px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#effeff,#45d7ef 40%,#08788f 80%);box-shadow:0 0 17px rgba(72,215,239,.65)}.orb-safe span{background:radial-gradient(circle at 35% 30%,#ecfdf5,#6ee7b7 40%,#047857 80%)}.ai-name{display:flex;align-items:center;gap:6px}.ai-name h2{margin:0;font-size:15px}.ai-name span{border:1px solid rgba(69,197,138,.24);border-radius:99px;padding:2px 5px;color:#72d5a1;font-size:8px;font-weight:700;text-transform:uppercase}.ai-head p{margin:3px 0 0;color:#718095;font-size:9px}.ai-head button{display:grid;width:38px;height:38px;margin-left:auto;place-items:center;border:1px solid rgba(255,255,255,.08);border-radius:6px;background:rgba(255,255,255,.025);color:#8090a0}.ai-head button svg{width:15px}.ai-sync{display:flex;align-items:center;gap:7px;margin-top:12px;border:1px solid rgba(72,215,239,.11);border-radius:6px;background:rgba(72,215,239,.035);padding:8px 9px;color:#8ca1b2;font-size:9px}.ai-sync svg{width:13px;color:#6bd9ed}.ai-sync b{margin-left:auto;color:#617386}
        .ai-head .speech-playing{border-color:rgba(72,215,239,.35);background:rgba(72,215,239,.08);color:#6de2f5}.ai-head .speech-error{border-color:rgba(240,82,136,.35);color:#f4779f}.speech-loading svg{animation:speech-spin .8s linear infinite}
        .ai-body{flex:1;min-height:0;overflow-y:auto;padding:17px 2px 12px 0;scrollbar-width:thin;scrollbar-color:rgba(72,215,239,.22) transparent}.ai-bubble{display:flex;gap:8px}.bot-mark{display:grid;width:28px;height:28px;place-items:center;border:1px solid rgba(72,215,239,.2);border-radius:6px;color:#70dcef}.bot-mark svg{width:14px}.ai-bubble>div:last-child{flex:1;border:1px solid rgba(255,255,255,.08);border-radius:4px 8px 8px 8px;background:#101923;padding:12px}.ai-bubble span,.ai-cause span,.ai-action-card span{color:#6c7c8d;font-size:8px;font-weight:700;letter-spacing:.09em;text-transform:uppercase}.ai-bubble p{margin:6px 0 0;color:#c6d1da;font-size:13px;line-height:1.58}.ai-cause{margin:12px 0 0 36px;border-left:2px solid rgba(72,215,239,.26);padding:2px 0 2px 11px}.ai-cause span{display:flex;align-items:center;gap:5px}.ai-cause svg{width:12px;color:#68d9ed}.ai-cause strong{display:block;margin-top:7px;color:#dce7ee;font-size:12px}.ai-cause p{margin:3px 0 0;color:#8596a7;font-size:10px}.ai-action-card{margin:15px 0 0 36px;border:1px solid rgba(72,215,239,.16);border-radius:7px;background:rgba(72,215,239,.045);padding:12px}.ai-action-card span{display:flex;align-items:center;gap:6px}.ai-action-card svg{width:13px;color:#6edcef}.ai-action-card strong{display:block;margin-top:8px;color:#e5f7fa;font-size:13px;line-height:1.4}.ai-action-card p{margin:6px 0 0;color:#7f9ca6;font-size:10px}.action-safe{border-color:rgba(69,197,138,.2);background:rgba(69,197,138,.05)}.action-safe strong,.action-safe svg{color:#76d9a5}
        .ai-conversation{display:grid;gap:7px;margin:12px 0 0 36px}.ai-message{border:1px solid rgba(255,255,255,.075);border-radius:6px;background:rgba(255,255,255,.025);padding:8px 9px}.ai-message>span{display:flex;align-items:center;justify-content:space-between;color:#758698;font-size:8px;font-weight:750;text-transform:uppercase}.ai-message span b{border:1px solid rgba(72,215,239,.2);border-radius:99px;padding:2px 5px;color:#64d5e9;font-size:7px}.ai-message p{margin:4px 0 0;color:#afbdc8;font-size:10px;line-height:1.45}.message-user{margin-left:18px;border-color:rgba(72,215,239,.14);background:rgba(72,215,239,.04)}.ai-thinking{display:flex;align-items:center;gap:6px;color:#7c8e9f;font-size:9px}.ai-thinking svg,.ai-input .send-loading svg{width:13px;animation:speech-spin .8s linear infinite}.ai-buttons{display:grid;border-top:1px solid rgba(255,255,255,.08);padding-top:12px}.ai-buttons button{display:inline-flex;min-height:42px;align-items:center;justify-content:center;gap:7px;border-radius:6px;font-size:10px;font-weight:700}.ai-buttons svg{width:14px}.accept{border:1px solid #22bfdc;background:#1397b1;color:#041318}.ai-buttons button:disabled{cursor:not-allowed;opacity:.42}.ai-input{display:grid;grid-template-columns:minmax(0,1fr) 38px 38px;gap:5px;margin-top:8px}.ai-input input{min-width:0;height:40px;border:1px solid rgba(255,255,255,.09);border-radius:6px;background:#0d151e;padding:0 10px;color:white;font-size:11px}.ai-input button{display:grid;width:38px;height:40px;place-items:center;border:1px solid rgba(255,255,255,.09);border-radius:6px;background:rgba(255,255,255,.03);color:#7790a1}.ai-input svg{width:14px}.ai-input button:disabled{cursor:not-allowed;opacity:.35}.ai-note{margin:7px 0 0;color:#566677;font-size:8px;text-align:center}
        .ai-input .mic-recording{border-color:rgba(240,82,136,.45);background:rgba(240,82,136,.1);color:#f4779f;box-shadow:0 0 14px rgba(240,82,136,.16)}.ai-input .mic-error{border-color:rgba(240,82,136,.35);color:#f4779f}.mic-transcribing svg{animation:speech-spin .8s linear infinite}

        /* Minimalist visual pass: semantic color stays, decorative effects recede. */
        .rg-shell{border-color:rgba(255,255,255,.07);border-radius:8px;background:#090d12;box-shadow:0 12px 36px rgba(0,0,0,.24)}
        .ambient-canvas{opacity:.12}
        .rg-header{border-color:rgba(255,255,255,.06)}
        .rg-kicker,.deck-caption span,.tower-rank,.gpu-rack-panel>header span,.gpu-detail-title span,.ai-bubble span,.ai-cause span,.ai-action-card span,.rg-stat span{letter-spacing:0}
        .rg-kicker{color:#8b9aa8}
        .rg-alert{border-color:rgba(240,100,100,.18);background:#10151c;color:#f07a7a}
        .rg-alert span{color:#788694}
        .layer-stage{border-color:rgba(255,255,255,.06);background:#0b1016}
        .deck-controls button,.ai-head button,.ai-input button{border-color:rgba(255,255,255,.07);background:#10161d}
        .deck-controls button:hover:not(:disabled),.ai-head button:hover,.ai-input button:hover:not(:disabled){background:#151d26}
        .tower{border-color:rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(69,197,138,.055) 0%,#111820 34%,#0d131a 100%);box-shadow:0 10px 24px rgba(0,0,0,.2)}
        .tower::before{display:none}
        .tower[aria-pressed=true]{border-color:rgba(220,232,238,.2);box-shadow:0 10px 26px rgba(0,0,0,.26)}
        .tower-critical[aria-pressed=true]{animation:none}
        .tower-gpu-count{color:#eef4f7}
        .tower-load{margin-top:auto}
        .tower-track{background:rgba(255,255,255,.07)}
        .tower-foot{border-color:rgba(255,255,255,.055)}
        .gpu-rack-panel{width:60%;border-left:0;background:#0b1118;padding:14px 16px 16px 20px;animation:none}
        .gpu-rack-panel>header.gpu-rack-header{display:grid;height:62px;grid-template-rows:28px 32px;gap:2px;align-items:center}.gpu-rack-title-row{display:flex;min-width:0;align-items:center;justify-content:space-between;gap:12px}.gpu-rack-title-row>div:first-child{min-width:0}.gpu-rack-title-row h2{margin:2px 0 0}.rack-back{display:inline-flex;width:max-content;min-height:28px;align-items:center;gap:5px;border:0;background:transparent;padding:0;color:#8998a6;font:inherit;font-size:9px;font-weight:700;cursor:pointer}.rack-back svg{width:13px}.rack-back:hover{color:#f2f6f8}.rack-back:focus-visible{outline:2px solid #43c7e8;outline-offset:2px}
        .gpu-card-grid{height:calc(100% - 70px);margin-top:8px;gap:8px}
        .gpu-panel-status{border-color:rgba(255,255,255,.07);background:#10161d}.gpu-panel-status.status-critical{border-color:rgba(255,255,255,.07);color:#e2777f}.gpu-panel-status.status-warning{border-color:rgba(255,255,255,.07);color:#d6a45d}.gpu-panel-status.status-normal{border-color:rgba(255,255,255,.07);color:#6fc598}
        .gpu-nav-card,.gpu-nav-card.gpu-critical,.gpu-nav-card.gpu-warning,.gpu-nav-card.gpu-normal{border-color:rgba(255,255,255,.065);background:linear-gradient(180deg,rgba(69,197,138,.045) 0%,#111820 38%,#0e151c 100%)}
        .gpu-nav-card:hover{border-color:rgba(220,232,238,.18);background:linear-gradient(180deg,rgba(69,197,138,.065) 0%,#141c24 42%,#10171e 100%);transform:none}
        .gpu-critical .gpu-nav-head>i,.gpu-critical .gpu-nav-temp{color:#e2777f}.gpu-warning .gpu-nav-head>i,.gpu-warning .gpu-nav-temp{color:#d6a45d}.gpu-normal .gpu-nav-head>i,.gpu-normal .gpu-nav-temp{color:#6fc598}
        .gpu-snapshot,.gpu-decision-block{border-color:rgba(255,255,255,.07);background:#10171f}
        .gpu-snapshot>span{background:#10171f}
        .rg-statusbar{margin-top:0;border-color:rgba(255,255,255,.06);background:#0d131a}
        .rg-ai{border-color:rgba(255,255,255,.06);background:#0b1016}
        .ai-head{border-color:rgba(255,255,255,.06)}
        .ai-orb{width:36px;height:36px;border-color:rgba(72,199,232,.25)}
        .ai-orb::before{display:none}
        .ai-orb span{width:9px;height:9px;background:#43c7e8;box-shadow:none}
        .orb-safe span{background:#45c58a}
        .ai-sync{border-color:rgba(255,255,255,.06);background:#10161d}
        .bot-mark{border-color:rgba(255,255,255,.07);color:#63cbe3}
        .ai-bubble>div:last-child{border:0;background:transparent;padding:4px 0}
        .ai-cause{border-color:rgba(255,255,255,.1)}
        .ai-action-card{border-color:rgba(255,255,255,.07);background:#10171f}
        .ai-message{border-color:rgba(255,255,255,.06);background:#10161d}
        .message-user{border-color:rgba(72,199,232,.16);background:#101820}
        .accept{border-color:#239eb8;background:#168da5;color:#061317}
        .ai-input input{border-color:rgba(255,255,255,.07);background:#10161d}
        .ai-input .mic-recording{box-shadow:none}

        @keyframes thermal-pulse{50%{box-shadow:0 0 0 1px rgba(240,82,136,.2),0 0 30px rgba(240,82,136,.32),inset 0 0 34px rgba(0,0,0,.36)}}@keyframes orb{50%{transform:scale(1.12);opacity:.45}}@keyframes speech-spin{to{transform:rotate(360deg)}}@keyframes gpu-panel-in{from{opacity:0;transform:translateX(8px)}}
        @media(max-height:760px) and (min-width:761px){
          .rg-ai{padding:14px 16px}.ai-head{padding-bottom:10px}.ai-orb{width:38px;height:38px}
          .ai-sync{margin-top:9px;padding:7px 9px}.ai-body{padding:11px 0 8px}
          .ai-bubble>div:last-child{padding:10px}.ai-bubble p{font-size:11px;line-height:1.45}
          .ai-cause{margin-top:8px}.ai-cause strong{margin-top:4px}.ai-action-card{margin-top:10px;padding:10px}
          .ai-action-card strong{margin-top:5px;font-size:11px}.ai-action-card p{margin-top:3px}
          .ai-buttons{padding-top:9px}.ai-buttons button{min-height:38px}.ai-input input,.ai-input button{height:36px}
          .ai-note{margin-top:4px}
        }
        @media(max-width:1000px){.rg-shell{grid-template-columns:1fr 330px}.deck-rack{width:160px;height:280px;transform:translate(-50%,-50%) translateX(calc(var(--deck-offset) * 166px)) scale(calc(1 - var(--deck-distance) * .1))}.with-gpu-grid .deck-rack{top:14px;width:112px;height:238px;transform:translate(-50%,0) translateX(calc(var(--deck-offset) * 102px)) scale(calc(1 - var(--deck-distance) * .12))}.with-gpu-grid .rack-deck{right:67%}.gpu-rack-panel{width:64%;padding-right:11px;padding-left:16px}.analysis-layer{grid-template-columns:1fr}}
        @media(max-width:760px){
          html,body{overflow:auto}.rg-shell{display:block;height:auto;min-height:100vh}.rg-ai{border-top:1px solid rgba(255,255,255,.09);border-left:0}
          .overview-layer,.analysis-layer,.simulation-layer{min-height:430px}.rg-statusbar{grid-template-columns:1fr 1fr}.layer-button kbd{display:none}
          .with-gpu-grid .rack-deck{display:none}
          .gpu-rack-panel{left:0;width:100%;padding:14px 16px 16px;overflow-y:auto}
          .gpu-card-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
          .gpu-detail>*{flex-shrink:0}
          .gpu-detail .gpu-details{overflow:visible}
        }
        @media(prefers-reduced-motion:reduce){.tower-critical[aria-pressed=true],.ai-orb::before{animation:none}.deck-rack{transition:none}}
      `}</style>
    </section>
  );
}

function mergeAudioChunks(chunks: Float32Array[]) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples: Float32Array, sampleRate: number) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

function AmbientCanvas({ state }: { state: RackState }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const particles = Array.from({ length: 34 }, (_, index) => ({
      x: (index * 83) % 1000,
      y: (index * 137) % 700,
      speed: .12 + (index % 5) * .04,
      radius: 1 + (index % 3) * .45
    }));
    let frame = 0;

    function resize() {
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
      canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
      context?.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    }
    function draw() {
      if (!canvas || !context) return;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      context.clearRect(0, 0, width, height);
      const color = state === "critical" ? "240,82,136" : state === "warning" ? "240,168,75" : "72,215,239";
      const gradient = context.createRadialGradient(width * .38, height * .42, 0, width * .38, height * .42, width * .55);
      gradient.addColorStop(0, `rgba(${color},.055)`);
      gradient.addColorStop(1, `rgba(${color},0)`);
      context.fillStyle = gradient;
      context.fillRect(0, 0, width, height);
      for (const particle of particles) {
        if (!reduced) particle.y = (particle.y - particle.speed + height) % height;
        context.beginPath();
        context.arc((particle.x / 1000) * width, particle.y % height, particle.radius, 0, Math.PI * 2);
        context.fillStyle = `rgba(${color},.16)`;
        context.fill();
      }
      if (!reduced) frame = requestAnimationFrame(draw);
    }
    resize();
    window.addEventListener("resize", resize);
    draw();
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
    };
  }, [state]);

  return <canvas ref={ref} className="ambient-canvas" aria-hidden="true" />;
}

function OverviewLayer({ activeIndex, selectedIndex, selectedGpuId, accepted, onMove, onSelect, onSelectGpu, onBackToRack, onBackToRacks, onWheel }: {
  activeIndex: number; selectedIndex: number | null; selectedGpuId: string | null; accepted: boolean; onMove: (direction: number) => void; onSelect: (index: number) => void;
  onSelectGpu: (gpuId: string) => void; onBackToRack: () => void; onBackToRacks: () => void;
  onWheel: (event: WheelEvent) => void;
}) {
  function openRack(index: number) {
    onSelect(index);
  }

  return (
    <div
      className={`overview-layer ${selectedIndex !== null ? "with-gpu-grid" : ""}`}
      onWheel={onWheel}
    >
      <div className="deck-caption"><span>Rack carousel</span><strong>{selectedIndex === null ? "Click a rack to inspect its 8 GPUs" : `${racks[selectedIndex].id} selected`}</strong></div>
      <div className="deck-controls">
        <button type="button" onClick={(event) => { event.stopPropagation(); onMove(-1); }} disabled={activeIndex === 0} aria-label="Previous rack"><ArrowLeft /></button>
        <button type="button" onClick={(event) => { event.stopPropagation(); onMove(1); }} disabled={activeIndex === racks.length - 1} aria-label="Next rack"><ArrowRight /></button>
      </div>
      <div className="rack-deck">
        {racks.map((rack, index) => {
          const offset = index - activeIndex;
          const distance = Math.abs(offset);
          const state = accepted && index === activeIndex ? "safe" : rack.state;
          const style = {
            ...palette[state],
            "--deck-offset": offset,
            "--deck-distance": distance,
            "--deck-opacity": distance > 3 ? 0 : Math.max(.25, 1 - distance * .22),
            "--deck-z": 10 - distance,
            "--deck-events": distance > 3 ? "none" : "auto"
          } as CSSProperties;
          return <DeckRack key={rack.id} rack={rack} rank={index + 1} state={state} active={selectedIndex === index || (selectedIndex === null && index === activeIndex)} style={style} onOpen={() => openRack(index)} />;
        })}
      </div>
      {selectedIndex !== null && (
        <GpuRackPanel
          rack={racks[selectedIndex]}
          selectedGpuId={selectedGpuId}
          onSelectGpu={onSelectGpu}
          onBack={onBackToRack}
          onBackToRacks={onBackToRacks}
        />
      )}
    </div>
  );
}

function DeckRack({ rack, rank, state, active, style, onOpen }: { rack: Rack; rank: number; state: RackState; active: boolean; style: CSSProperties; onOpen: () => void }) {
  const Icon = state === "critical" ? CircleAlert : state === "warning" ? TriangleAlert : CheckCircle2;
  return (
    <div className="deck-rack" style={style}>
      <button
        className={`tower tower-${state}`}
        aria-pressed={active}
        title={`Open ${rack.id}`}
        onClick={onOpen}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onOpen();
          }
        }}
      >
        <span className="tower-rank">Priority {String(rank).padStart(2, "0")}</span>
        <div className="tower-head"><Server /><strong>{rack.id}</strong><i><Icon /></i></div>
        <div className="tower-gpu-count"><strong>{rack.gpu_count}</strong><span>GPUs</span></div>
        <div className="tower-risk-counts">
          <span><b>{rack.normal_count}</b> normal</span>
          <span><b>{rack.warning_count}</b> warning</span>
          <span><b>{rack.critical_count}</b> critical</span>
        </div>
        <div className="tower-load"><div className="tower-track"><span style={{ width: `${(rack.gpus_to_check / rack.gpu_count) * 100}%` }} /></div><b>{rack.gpus_to_check} to check</b></div>
        <div className="tower-foot"><span>Rack status</span><strong>{rack.status}</strong></div>
      </button>
    </div>
  );
}

function GpuRackPanel({ rack, selectedGpuId, onSelectGpu, onBack, onBackToRacks }: {
  rack: Rack;
  selectedGpuId: string | null;
  onSelectGpu: (gpuId: string) => void;
  onBack: () => void;
  onBackToRacks: () => void;
}) {
  const selectedGpu = rack.gpus.find((gpu) => gpu.gpu_id === selectedGpuId);

  if (selectedGpu) {
    return <GpuDetail rack={rack} gpu={selectedGpu} onBack={onBack} />;
  }

  return (
    <section className="gpu-rack-panel" aria-labelledby="gpu-rack-title">
      <header className="gpu-rack-header">
        <button type="button" className="rack-back" onClick={onBackToRacks}><ArrowLeft />Back to racks</button>
        <div className="gpu-rack-title-row">
          <div>
            <span>Selected rack</span>
            <h2 id="gpu-rack-title">{rack.id} · {rack.gpu_count} GPUs</h2>
          </div>
          <div className={`gpu-panel-status status-${rack.status}`}>
            {rack.gpus_to_check} to check
          </div>
        </div>
      </header>
      <div className="gpu-card-grid">
        {rack.gpus.map((gpu) => (
          <GpuCard key={gpu.gpu_id} gpu={gpu} onClick={() => onSelectGpu(gpu.gpu_id)} />
        ))}
      </div>
    </section>
  );
}

function GpuCard({ gpu, onClick }: { gpu: GpuSnapshot; onClick: () => void }) {
  const Icon = gpu.status === "critical" ? CircleAlert : gpu.status === "warning" ? TriangleAlert : CheckCircle2;
  return (
    <button type="button" className={`gpu-nav-card gpu-${gpu.status}`} onClick={onClick}>
      <div className="gpu-nav-head">
        <span><Cpu />{gpu.gpu_id}</span>
        <i><Icon />{gpu.status}</i>
      </div>
      <div className="gpu-nav-temp">{gpu.temperature_c}<small>°C</small></div>
      <dl>
        <div><dt>Predicted</dt><dd>{gpu.predicted_temperature_c}°C</dd></div>
        <div><dt>Utilization</dt><dd>{gpu.utilization_percent}%</dd></div>
        <div><dt>Power</dt><dd>{gpu.power_draw_w} W</dd></div>
      </dl>
    </button>
  );
}

function GpuDetail({ rack, gpu, onBack }: { rack: Rack; gpu: GpuSnapshot; onBack: () => void }) {
  const [decision, setDecision] = useState<GpuDecisionData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    setLoading(true);
    setDecision(null);

    void loadGpuDecision(gpu, controller.signal).then((result) => {
      if (!current) return;
      setDecision(result);
      setLoading(false);
    });

    return () => {
      current = false;
      controller.abort();
    };
  }, [gpu]);

  const prediction = decision?.prediction;
  const recommendation = decision?.recommendation;
  const observed = decision?.current;
  const roi = decision?.roi;
  const diagnosis = recommendation?.diagnosis?.likely_cause ?? gpu.risk_reason;
  const why = recommendation?.why?.length
    ? recommendation.why.join(" ")
    : gpu.risk_reason;
  const action = recommendation?.action_label ?? gpu.recommended_action;
  const riskScore =
    recommendation?.thermal_context?.risk_no_action ?? gpu.risk_score;
  const riskPercent = Math.round(riskScore * 100);
  const currentTemp = observed?.current_temp_c ?? gpu.temperature_c;
  const predictedTemp =
    prediction?.predicted_equilibrium_temp_c ??
    prediction?.predicted_peak_temp_c ??
    gpu.predicted_temperature_c;
  const utilization = observed?.utilization_percent ?? gpu.utilization_percent;
  const power = observed?.power_draw_w ?? gpu.power_draw_w;
  const displayStatus =
    prediction?.risk === "safe" ? "normal" : prediction?.risk ?? gpu.status;

  return (
    <section className="gpu-rack-panel gpu-detail" aria-labelledby="gpu-detail-title">
      <header>
        <button type="button" className="gpu-back" onClick={onBack}><ArrowLeft />Back to rack</button>
        <div className="gpu-detail-status">
          {loading && <span className="gpu-loading"><LoaderCircle />Syncing</span>}
          {!loading && decision && <span className="gpu-source">{decision.source === "backend" ? "Live backend" : "Demo fallback"}</span>}
          <div className={`gpu-panel-status status-${displayStatus}`}>{displayStatus}</div>
        </div>
      </header>
      <div className="gpu-detail-title">
        <span>{rack.id} / {gpu.gpu_id}</span>
        <h2 id="gpu-detail-title">GPU IA Analysis</h2>
      </div>

      <div className="gpu-snapshot">
        <span><small>Current</small><strong>{currentTemp ?? "N/A"}{currentTemp != null && "°C"}</strong></span>
        <span><small>Predicted</small><strong>{predictedTemp ?? "N/A"}{predictedTemp != null && "°C"}</strong></span>
        <span><small>Utilization</small><strong>{utilization ?? "N/A"}{utilization != null && "%"}</strong></span>
        <span><small>Power</small><strong>{power ?? "N/A"}{power != null && " W"}</strong></span>
      </div>

      <div className="gpu-decision-block">
        <div className="gpu-block-heading"><Sparkles /><span><small>IA Analysis</small><strong>{riskPercent}% risk · {displayStatus}</strong></span></div>
        <p><b>Diagnosis:</b> {humanizeBackendText(diagnosis)}</p>
        <p><b>Why:</b> {humanizeBackendText(why)}</p>
        <div className="gpu-recommendation"><Lightbulb /><span><small>Recommended action</small><strong>{action}</strong></span></div>
      </div>

      <details className="gpu-details">
        <summary>Details</summary>
        <div className="gpu-details-content">
          <section>
            <small>Forecast</small>
            <p>Peak {formatValue(prediction?.predicted_peak_temp_c, "°C")} · Safe limit {formatValue(prediction?.safe_limit_c, "°C")} · Confidence {formatPercent(prediction?.confidence)}</p>
          </section>
          {roi && (
            <section>
              <small>ROI</small>
              <p>Net gain {formatCurrency(roi.net_gain_eur)} · ROI {roi.roi_percent == null ? "N/A" : `${Math.round(roi.roi_percent)}%`} · {humanizeBackendText(roi.decision)}</p>
            </section>
          )}
          {(recommendation?.action_scores?.length ?? 0) > 0 && (
            <section>
              <small>Action scores</small>
              <ol>
                {recommendation!.action_scores!.slice(0, 3).map((candidate, index) => (
                  <li key={candidate.action_id ?? candidate.action_type ?? index}>
                    <span>
                      <strong>{candidate.label ?? humanizeBackendText(candidate.action_type ?? candidate.action_id ?? "Action")}</strong>
                      <small>
                        {candidate.cooling_gain_c != null && `${candidate.cooling_gain_c}°C cooling`}
                        {candidate.risk_reduction != null && ` · ${Math.round(candidate.risk_reduction * 100)}% risk reduction`}
                        {(candidate.operational_cost || candidate.performance_impact) && ` · ${candidate.operational_cost ?? `${candidate.performance_impact} performance impact`}`}
                      </small>
                    </span>
                    {candidate.score != null && <b>{Math.round(candidate.score)}</b>}
                  </li>
                ))}
              </ol>
            </section>
          )}
          {recommendation?.report && (
            <section>
              <small>Report</small>
              <p>{recommendation.report.incident_summary ?? recommendation.llm_report?.summary ?? "Not provided"}</p>
            </section>
          )}
          {recommendation?.migration_plan && (
            <section>
              <small>Prepare Migration Plan</small>
              <p><b>Simulated plan:</b> {recommendation.migration_plan.target_action}</p>
              <p>{humanizeBackendText(recommendation.migration_plan.reason)}</p>
              <ol>
                {recommendation.migration_plan.steps.map((step, index) => (
                  <li key={`${step}-${index}`}><span>{index + 1}. {step}</span></li>
                ))}
              </ol>
              <p>Migration plan is simulated and operator-ready. It does not execute infrastructure changes automatically. Operator confirmation is required.</p>
            </section>
          )}
        </div>
      </details>
    </section>
  );
}

function humanizeBackendText(value: string) {
  return value.replaceAll("_", " ").replace(/^\w/, (letter) => letter.toUpperCase());
}

function formatValue(value: number | null | undefined, suffix: string) {
  return value == null ? "N/A" : `${Math.round(value * 10) / 10}${suffix}`;
}

function formatPercent(value: number | null | undefined) {
  if (value == null) return "N/A";
  return `${Math.round(value <= 1 ? value * 100 : value)}%`;
}

function formatCurrency(value: number | null | undefined) {
  if (value == null) return "N/A";
  return new Intl.NumberFormat("en-IE", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 2,
  }).format(value);
}

function Stat({ icon: Icon, label, value, accent = false }: { icon: typeof Cpu; label: string; value: string; accent?: boolean }) {
  return <div className={`rg-stat ${accent ? "accent" : ""}`}><Icon /><div><span>{label}</span><strong>{value}</strong></div></div>;
}
