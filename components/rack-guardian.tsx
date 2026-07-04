"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  Gauge,
  Info,
  Mic,
  Radio,
  ShieldCheck,
  Sparkles,
  Speaker,
  Wind
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

const copy = {
  alert:
    "Rack R-04 is converging toward an unsafe temperature. Slowdown expected in 6 minutes. I recommend moving the workload to Rack R-06.",
  why:
    "The workload has drawn unusually high power for several minutes. That power becomes heat. With cooling flow also reduced, the rack cannot return to a safe equilibrium.",
  accepted:
    "Action accepted. Workload migration simulated. Rack R-04 is now converging toward 74°C. Slowdown avoided."
};

export function RackGuardian() {
  const [accepted, setAccepted] = useState(false);
  const [message, setMessage] = useState<string>(copy.alert);
  const [typedMessage, setTypedMessage] = useState("");
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const speechSupported = useMemo(
    () => typeof window !== "undefined" && "speechSynthesis" in window,
    []
  );

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setTypedMessage(message);
      return;
    }
    setTypedMessage("");
    let index = 0;
    const timer = window.setInterval(() => {
      index += 2;
      setTypedMessage(message.slice(0, index));
      if (index >= message.length) window.clearInterval(timer);
    }, 26);
    return () => window.clearInterval(timer);
  }, [message]);

  useEffect(() => () => window.speechSynthesis?.cancel(), []);

  function speak(text: string) {
    if (!speechSupported) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.94;
    utterance.pitch = 0.96;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }

  function acceptRecommendation() {
    setAccepted(true);
    setMessage(copy.accepted);
    if (voiceEnabled) speak(copy.accepted);
  }

  function askWhy() {
    setMessage(copy.why);
    if (voiceEnabled) speak(copy.why);
  }

  return (
    <main className={cn("app-shell", accepted && "is-safe")}>
      <div className="ambient-grid" aria-hidden="true" />
      <div className="mx-auto min-h-screen w-full max-w-[1180px] px-4 py-4 sm:px-6 sm:py-5">
        <Header safe={accepted} />

        <section className="decision-layout">
          <div className="assistant-stage">
            <AssistantIdentity speaking={speaking} safe={accepted} />
            <div className="assistant-copy">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-200/70">
                  <Sparkles className="h-4 w-4" aria-hidden="true" />
                  Co-pilot assessment
                </div>
                <StatusPill safe={accepted} compact />
              </div>

              <p className="assistant-message" aria-live="polite">
                {typedMessage}
                {typedMessage.length < message.length && <span className="typing-cursor" />}
              </p>

              <div className={cn("outcome-strip", accepted && "outcome-safe")}>
                {accepted ? <ShieldCheck aria-hidden="true" /> : <ArrowRight aria-hidden="true" />}
                <div>
                  <span>{accepted ? "Safe outcome" : "Expected outcome"}</span>
                  <strong>
                    {accepted ? "Thermal slowdown avoided" : "90°C → 74°C · Slowdown avoided"}
                  </strong>
                </div>
              </div>
            </div>
          </div>

          <OperatorControls
            accepted={accepted}
            detailsOpen={detailsOpen}
            speaking={speaking}
            speechSupported={speechSupported}
            onAccept={acceptRecommendation}
            onAskWhy={askWhy}
            onToggleDetails={() => setDetailsOpen((value) => !value)}
            onSpeak={() => {
              setVoiceEnabled(true);
              speak(message);
            }}
          />

          <div className="evidence-grid">
            <AlertSummary safe={accepted} />
            <Recommendation safe={accepted} />
          </div>

          {detailsOpen && <TechnicalDetails safe={accepted} />}
        </section>
      </div>
    </main>
  );
}

function Header({ safe }: { safe: boolean }) {
  return (
    <header className="topbar">
      <div className="flex items-center gap-3">
        <div className="brand-symbol" aria-hidden="true">
          <Activity className="h-5 w-5" />
        </div>
        <div>
          <p className="text-[15px] font-semibold text-white sm:text-base">Rack Guardian</p>
          <p className="text-xs text-slate-500">AI thermal co-pilot</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className="hidden items-center gap-2 text-xs text-slate-500 sm:flex">
          <Radio className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
          Live simulation
        </span>
        <StatusPill safe={safe} />
      </div>
    </header>
  );
}

function StatusPill({ safe, compact = false }: { safe: boolean; compact?: boolean }) {
  return (
    <span className={cn("status-pill", safe ? "status-safe" : "status-critical", compact && "status-compact")}>
      {safe ? <ShieldCheck aria-hidden="true" /> : <CircleAlert aria-hidden="true" />}
      {safe ? "Safe" : "Critical"}
    </span>
  );
}

function AssistantIdentity({ speaking, safe }: { speaking: boolean; safe: boolean }) {
  return (
    <div className="assistant-identity">
      <div
        className={cn("orb", safe && "orb-safe", speaking && "orb-speaking")}
        role="img"
        aria-label={speaking ? "Assistant speaking" : "Assistant monitoring"}
      >
        <span className="orb-halo" />
        <span className="orb-ring" />
        <span className="orb-core" />
      </div>
      <div className="presence">
        <span />
        <div>
          <strong>Monitoring cluster</strong>
          <small>12 racks online</small>
        </div>
      </div>
      <div className="waveform" aria-hidden="true">
        {[7, 15, 10, 21, 12, 17, 8, 14, 6].map((height, index) => (
          <span key={index} style={{ height }} />
        ))}
      </div>
    </div>
  );
}

type ControlsProps = {
  accepted: boolean;
  detailsOpen: boolean;
  speaking: boolean;
  speechSupported: boolean;
  onAccept: () => void;
  onAskWhy: () => void;
  onToggleDetails: () => void;
  onSpeak: () => void;
};

function OperatorControls(props: ControlsProps) {
  return (
    <div className="controls">
      <Button
        variant="primary"
        className={cn("primary-action", props.accepted && "accepted-action")}
        onClick={props.onAccept}
        disabled={props.accepted}
      >
        {props.accepted ? <Check aria-hidden="true" /> : <ShieldCheck aria-hidden="true" />}
        {props.accepted ? "Recommendation accepted" : "Accept recommendation"}
      </Button>
      <Button onClick={props.onAskWhy}>
        <Info aria-hidden="true" />
        Ask why
      </Button>
      <AlternativesDialog />
      <div className="control-spacer" />
      <Button
        variant="ghost"
        onClick={props.onSpeak}
        disabled={!props.speechSupported || props.speaking}
        title={props.speechSupported ? "Read the current message aloud" : "Speech unavailable"}
      >
        <Speaker aria-hidden="true" />
        {props.speaking ? "Speaking…" : "Speak alert"}
      </Button>
      <Button
        variant="ghost"
        size="icon"
        disabled
        title="Voice commands coming soon"
        aria-label="Voice commands coming soon"
      >
        <Mic aria-hidden="true" />
      </Button>
      <Button variant="ghost" onClick={props.onToggleDetails}>
        Technical details
        {props.detailsOpen ? <ChevronUp aria-hidden="true" /> : <ChevronDown aria-hidden="true" />}
      </Button>
    </div>
  );
}

function AlertSummary({ safe }: { safe: boolean }) {
  return (
    <article className="evidence-card" aria-labelledby="rack-summary">
      <div className="card-heading">
        <div>
          <p>Active alert</p>
          <h2 id="rack-summary">Rack R-04</h2>
        </div>
        <StatusPill safe={safe} />
      </div>
      <div className="temperature-row">
        <div>
          <span>Predicted equilibrium</span>
          <strong className={safe ? "temperature-safe" : "temperature-critical"}>
            {safe ? 74 : 90}°C
          </strong>
        </div>
        <div className="current-temperature">
          <span>Current</span>
          <strong>84°C</strong>
        </div>
      </div>
      <div className="metrics">
        <Metric label="Safe limit" value="85°C" />
        <Metric label="Slowdown" value={safe ? "Avoided" : "6 min 40 sec"} />
      </div>
    </article>
  );
}

function Recommendation({ safe }: { safe: boolean }) {
  return (
    <article className={cn("evidence-card recommendation-card", safe && "recommendation-safe")}>
      <div className="card-heading">
        <div>
          <p>{safe ? "Completed action" : "Recommended action"}</p>
          <h2>{safe ? "Workload migration simulated" : "Move llm-ft-2841 to Rack R-06"}</h2>
        </div>
        <Gauge className={cn("h-5 w-5", safe ? "text-emerald-300" : "text-cyan-300")} aria-hidden="true" />
      </div>
      <p className="explanation">
        {safe
          ? "Rack R-04 is returning to a safe thermal equilibrium. No throttling is expected."
          : "The rack is producing more heat than cooling can remove. High GPU power draw is the main cause, with reduced airflow as a secondary factor."}
      </p>
      <div className={cn("recommendation-result", safe && "result-safe")}>
        <Check aria-hidden="true" />
        <span>{safe ? "Converging toward 74°C" : "Temperature drops by 16°C · Slowdown avoided"}</span>
      </div>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AlternativesDialog() {
  const options = [
    { label: "No action", temperature: "90°C", outcome: "Slowdown likely", risk: true },
    { label: "Increase cooling", temperature: "79°C", outcome: "Safe", risk: false },
    { label: "Reduce GPU frequency", temperature: "76°C", outcome: "Safe", risk: false }
  ];

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>
          <Wind aria-hidden="true" />
          Show alternatives
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle className="text-xl font-semibold text-white">Alternative responses</DialogTitle>
        <DialogDescription className="mt-2 text-sm leading-6 text-slate-400">
          Compare simple outcomes before making an operational decision.
        </DialogDescription>
        <div className="mt-6 space-y-2">
          {options.map((option) => (
            <div className="alternative" key={option.label}>
              <div>
                <strong>{option.label}</strong>
                <span className={option.risk ? "text-red-300" : "text-emerald-300"}>{option.outcome}</span>
              </div>
              <b className={option.risk ? "text-red-300" : "text-emerald-300"}>{option.temperature}</b>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function TechnicalDetails({ safe }: { safe: boolean }) {
  return (
    <section className="technical-details" aria-label="Technical details">
      <div className="detail-metrics">
        <Metric label="Current temperature" value="84°C" />
        <Metric label="Predicted equilibrium" value={safe ? "74°C" : "90°C"} />
        <Metric label="Before throttling" value={safe ? "Avoided" : "6 min 40 sec"} />
        <Metric label="Confidence" value="92%" />
      </div>
      <div className="technical-grid">
        <div>
          <p className="technical-label">Temperature projection</p>
          <svg viewBox="0 0 560 130" role="img" aria-label="Predicted rack temperature curve">
            <line x1="12" y1="52" x2="548" y2="52" stroke="#405064" strokeDasharray="5 8" />
            <text x="14" y="43" fill="#718096" fontSize="11">85°C safe limit</text>
            <path
              d={safe ? "M12 98 C120 74 185 62 265 74 S430 100 548 105" : "M12 98 C130 83 188 62 275 48 S430 26 548 21"}
              fill="none"
              stroke={safe ? "#45c58a" : "#f06464"}
              strokeWidth="3"
              strokeLinecap="round"
            />
          </svg>
        </div>
        <div>
          <p className="technical-label">Contributing factors</p>
          <Cause label="GPU power draw" value={68} />
          <Cause label="Reduced cooling flow" value={24} />
          <Cause label="Ambient temperature" value={8} />
        </div>
      </div>
    </section>
  );
}

function Cause({ label, value }: { label: string; value: number }) {
  return (
    <div className="cause">
      <div><span>{label}</span><b>{value}%</b></div>
      <div className="cause-track"><span style={{ width: `${value}%` }} /></div>
    </div>
  );
}
