"use client";

import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent,
  useEffect,
  useRef,
  useState
} from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Bot,
  Check,
  CheckCircle2,
  CircleAlert,
  Cpu,
  Gauge,
  Layers3,
  Lightbulb,
  Mic,
  Play,
  Radio,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  Speaker,
  Thermometer,
  TriangleAlert
} from "lucide-react";

type RackState = "safe" | "warning" | "critical";
type View = "overview" | "analysis" | "simulation";

type Rack = {
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

const racks: Rack[] = [
  { id: "R-04", temperature: 84, predicted: 90, load: 96, safeLimit: 85, slowdown: "6 min 40 sec", state: "critical", cause: "High GPU power draw", secondaryCause: "Cooling flow reduced", recommendation: "Move workload llm-ft-2841 to Rack R-06." },
  { id: "R-14", temperature: 82, predicted: 87, load: 91, safeLimit: 85, slowdown: "14 min", state: "critical", cause: "Warm cooling intake", secondaryCause: "High sustained load", recommendation: "Increase cooling flow by 12% for this aisle." },
  { id: "R-10", temperature: 78, predicted: 84, load: 84, safeLimit: 85, slowdown: "31 min", state: "warning", cause: "Ambient temperature rising", secondaryCause: "Cooling margin narrowing", recommendation: "Monitor intake air and reassess in 10 minutes." },
  { id: "R-03", temperature: 75, predicted: 81, load: 79, safeLimit: 85, slowdown: "None", state: "warning", cause: "Sustained GPU workload", secondaryCause: "Normal cooling flow", recommendation: "Keep workload stable and monitor cooling flow." },
  { id: "R-08", temperature: 71, predicted: 76, load: 72, safeLimit: 85, slowdown: "None", state: "warning", cause: "Moderate load increase", secondaryCause: "Stable cooling capacity", recommendation: "No immediate action. Continue monitoring." },
  { id: "R-02", temperature: 68, predicted: 72, load: 63, safeLimit: 85, slowdown: "None", state: "safe", cause: "Balanced heat output", secondaryCause: "Healthy cooling reserve", recommendation: "No action needed." },
  { id: "R-07", temperature: 65, predicted: 69, load: 55, safeLimit: 85, slowdown: "None", state: "safe", cause: "Moderate utilization", secondaryCause: "Comfortable thermal margin", recommendation: "No action needed." },
  { id: "R-06", temperature: 62, predicted: 66, load: 42, safeLimit: 85, slowdown: "None", state: "safe", cause: "Low utilization", secondaryCause: "Spare cooling capacity", recommendation: "Available as a safe migration destination." }
];

const palette: Record<RackState, CSSProperties> = {
  safe: { "--accent": "#48d7ef", "--glow": "rgba(72,215,239,.22)", "--fill": "rgba(30,105,121,.22)" } as CSSProperties,
  warning: { "--accent": "#f0a84b", "--glow": "rgba(240,168,75,.28)", "--fill": "rgba(119,72,24,.24)" } as CSSProperties,
  critical: { "--accent": "#f05288", "--glow": "rgba(240,82,136,.42)", "--fill": "rgba(119,26,67,.28)" } as CSSProperties
};

export default function DataCenterRoom() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [view, setView] = useState<View>("overview");
  const [simulated, setSimulated] = useState(false);
  const [accepted, setAccepted] = useState(false);
  const [question, setQuestion] = useState("");
  const [aiReply, setAiReply] = useState<string | null>(null);
  const wheelLocked = useRef(false);

  const rack = racks[activeIndex];
  const outcome = rack.id === "R-04" ? 74 : Math.max(65, rack.predicted - 7);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
      if (event.key === "1") setView("overview");
      if (event.key === "2") setView("analysis");
      if (event.key === "3") setView("simulation");
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  function move(direction: number) {
    setActiveIndex((current) => Math.max(0, Math.min(racks.length - 1, current + direction)));
    setSimulated(false);
    setAccepted(false);
    setAiReply(null);
  }

  function select(index: number) {
    setActiveIndex(index);
    setSimulated(false);
    setAccepted(false);
    setAiReply(null);
  }

  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    if (wheelLocked.current || Math.abs(event.deltaY) < 8) return;
    wheelLocked.current = true;
    move(event.deltaY > 0 ? 1 : -1);
    window.setTimeout(() => { wheelLocked.current = false; }, 280);
  }

  function runSimulation() {
    setSimulated(true);
    setView("simulation");
  }

  function speakAnalysis() {
    if (!("speechSynthesis" in window)) return;
    const text = accepted
      ? `Action accepted. ${rack.id} is now converging toward ${outcome} degrees.`
      : `${rack.id} is predicted to reach ${rack.predicted} degrees. ${rack.recommendation}`;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }

  function askGuardian() {
    if (!question.trim()) return;
    const lower = question.toLowerCase();
    setAiReply(
      lower.includes("why") || lower.includes("pourquoi")
        ? `${rack.cause}. ${rack.secondaryCause}.`
        : lower.includes("solution") || lower.includes("action")
          ? rack.recommendation
          : `For ${rack.id}, the predicted equilibrium is ${rack.predicted}°C and the safe limit is ${rack.safeLimit}°C.`
    );
    setQuestion("");
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
          <div className="rg-alert"><CircleAlert /><div><strong>2 thermal risks</strong><span>R-04 needs action first</span></div></div>
        </header>

        <nav className="layer-tabs" aria-label="Workspace layers">
          <LayerButton active={view === "overview"} onClick={() => setView("overview")} icon={Layers3} label="Overview" shortcut="1" />
          <LayerButton active={view === "analysis"} onClick={() => setView("analysis")} icon={Sparkles} label="AI Analysis" shortcut="2" />
          <LayerButton active={view === "simulation"} onClick={() => setView("simulation")} icon={Play} label="Simulation" shortcut="3" />
          <div className="layer-context"><span>Selected</span><strong>{rack.id}</strong></div>
        </nav>

        <div className="layer-stage">
          {view === "overview" && (
            <OverviewLayer
              activeIndex={activeIndex}
              accepted={accepted}
              onMove={move}
              onSelect={select}
              onWheel={handleWheel}
            />
          )}
          {view === "analysis" && <AnalysisLayer rack={rack} />}
          {view === "simulation" && (
            <SimulationLayer rack={rack} outcome={outcome} simulated={simulated} accepted={accepted} onRun={() => setSimulated(true)} />
          )}
        </div>

        <div className="rg-statusbar" style={palette[accepted ? "safe" : rack.state]}>
          <Stat icon={Thermometer} label="Current" value={`${rack.temperature}°C`} />
          <Stat icon={Gauge} label={simulated || accepted ? "Outcome" : "Equilibrium"} value={`${simulated || accepted ? outcome : rack.predicted}°C`} accent />
          <Stat icon={ShieldCheck} label="Safe limit" value={`${rack.safeLimit}°C`} />
          <Stat icon={Activity} label="Slowdown" value={accepted ? "Avoided" : rack.slowdown} />
        </div>
      </div>

      <aside className="rg-ai">
        <header className="ai-head">
          <div className={`ai-orb ${accepted ? "orb-safe" : ""}`}><span /></div>
          <div><div className="ai-name"><h2>Guardian AI</h2><span>Live</span></div><p>Decision co-pilot</p></div>
          <button type="button" aria-label="Speak analysis" onClick={speakAnalysis}><Speaker /></button>
        </header>

        <div className="ai-sync"><Cpu /><span>Context synced with {rack.id}</span><b>#{activeIndex + 1}</b></div>

        <div className="ai-body" aria-live="polite">
          <div className="ai-bubble">
            <div className="bot-mark"><Bot /></div>
            <div>
              <span>Assessment</span>
              <p>
                {accepted
                  ? `Action accepted. ${rack.id} is now converging toward ${outcome}°C. Slowdown avoided.`
                  : rack.state === "critical"
                    ? `${rack.id} is converging toward an unsafe equilibrium of ${rack.predicted}°C. Action is required in ${rack.slowdown}.`
                    : rack.state === "warning"
                      ? `${rack.id} is warming but remains below the safe limit. There is time to monitor before intervening.`
                      : `${rack.id} is thermally stable with ${rack.safeLimit - rack.predicted}°C of predicted headroom.`}
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
          {aiReply && <div className="ai-reply"><span>Answer</span><p>{aiReply}</p></div>}
        </div>

        <div className="ai-buttons">
          <button className="simulate" type="button" onClick={runSimulation} disabled={accepted || rack.state === "safe"}><Play />{simulated ? "Simulated" : "Simulate"}</button>
          <button className="accept" type="button" onClick={() => setAccepted(true)} disabled={accepted || rack.state === "safe"}>{accepted ? <Check /> : <ShieldCheck />}{accepted ? "Accepted" : "Accept recommendation"}</button>
        </div>
        <div className="ai-input">
          <input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") askGuardian(); }} placeholder={`Ask about ${rack.id}…`} aria-label="Ask Guardian AI" />
          <button type="button" disabled aria-label="Voice commands coming soon"><Mic /></button>
          <button type="button" aria-label="Send message" onClick={askGuardian} disabled={!question.trim()}><Send /></button>
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
        .rg-main { display: grid; grid-template-rows: auto auto minmax(0,1fr) auto; min-width: 0; padding: 18px 22px; }
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
        .overview-layer { position: relative; width: 100%; height: 100%; min-height: 390px; overflow: hidden; touch-action: pan-y; user-select:none; cursor:grab; }
        .overview-layer.dragging{cursor:grabbing}
        .deck-caption { position:absolute;top:16px;left:18px;z-index:5 }.deck-caption span,.deck-caption strong{display:block}.deck-caption span{color:#657587;font-size:9px;letter-spacing:.1em;text-transform:uppercase}.deck-caption strong{margin-top:3px;font-size:13px}
        .deck-controls{position:absolute;top:14px;right:16px;z-index:8;display:flex;gap:6px}.deck-controls button{display:grid;width:38px;height:38px;place-items:center;border:1px solid rgba(255,255,255,.09);border-radius:6px;background:rgba(255,255,255,.03);color:#8292a2}.deck-controls button:hover:not(:disabled){background:rgba(255,255,255,.07);color:#fff}.deck-controls button:disabled{cursor:not-allowed;opacity:.28}.deck-controls svg{width:15px}
        .rack-deck { position:absolute;inset:56px 0 0;perspective:1000px; }
        .deck-rack { position:absolute;left:50%;top:50%;width:180px;height:310px;opacity:var(--deck-opacity);transform:translate(-50%,-50%) translateX(calc(var(--deck-offset) * 196px + var(--drag-x, 0px))) translateZ(calc((1 - var(--deck-distance)) * 55px)) rotateY(calc(var(--deck-offset) * -4deg)) scale(calc(1 - var(--deck-distance) * .09));z-index:var(--deck-z);transition:transform .32s cubic-bezier(.22,.8,.25,1),opacity .25s;pointer-events:var(--deck-events)}
        .overview-layer.dragging .deck-rack{transition:none}
        .tower { position:relative;display:flex;width:100%;height:100%;flex-direction:column;border:1px solid color-mix(in srgb,var(--accent) 38%,transparent);border-radius:5px 5px 8px 8px;background:repeating-linear-gradient(0deg,transparent 0 15px,rgba(255,255,255,.065) 15px 16px),linear-gradient(145deg,var(--fill),rgba(10,16,24,.98) 48%);padding:16px 14px;color:#edf6fa;text-align:left;box-shadow:inset 0 0 34px rgba(0,0,0,.4),0 18px 35px rgba(0,0,0,.3)}
        .tower::before{content:"";position:absolute;top:-8px;right:5px;left:5px;height:8px;border:1px solid color-mix(in srgb,var(--accent) 28%,transparent);border-bottom:0;background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 15%,#111a25),#090f16);clip-path:polygon(6% 100%,0 40%,10% 0,92% 0,100% 50%,94% 100%)}
        .tower[aria-pressed=true]{box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 25%,transparent),0 0 36px var(--glow),0 22px 38px rgba(0,0,0,.4)}.tower-critical[aria-pressed=true]{animation:thermal-pulse 2.8s ease-in-out infinite}
        .tower-rank{color:#6b7b8c;font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.tower-head{display:flex;align-items:center;gap:7px;margin-top:12px}.tower-head svg{width:14px}.tower-head strong{font-size:15px}.tower-head i{margin-left:auto;color:var(--accent)}.tower-temp{margin-top:22px;color:var(--accent);font-size:42px;font-weight:750;line-height:1;text-shadow:0 0 18px var(--glow)}.tower-temp small{font-size:14px;color:#8796a6}.tower-slots{display:grid;gap:6px;margin-top:auto;padding-top:20px}.tower-slots span{position:relative;height:6px;border-radius:1px;background:rgba(255,255,255,.07)}.tower-slots span::before{content:"";position:absolute;top:2px;left:4px;width:2px;height:2px;border-radius:50%;background:var(--accent);box-shadow:0 0 5px var(--accent)}.tower-load{display:flex;align-items:center;gap:8px;margin-top:13px}.tower-track{flex:1;height:4px;border-radius:4px;background:rgba(255,255,255,.08);overflow:hidden}.tower-track span{display:block;height:100%;background:var(--accent)}.tower-load b{color:#7c8b9b;font-size:9px}.tower-foot{display:flex;justify-content:space-between;margin-top:12px;border-top:1px solid rgba(255,255,255,.07);padding-top:9px;color:#68798a;font-size:9px}.tower-foot strong{color:var(--accent);font-size:11px}
        .analysis-layer,.simulation-layer{display:grid;width:100%;height:100%;min-height:390px;padding:22px}.analysis-layer{grid-template-columns:1.1fr .9fr;gap:16px}.analysis-summary,.cause-panel,.simulation-before,.simulation-after{border:1px solid rgba(255,255,255,.08);border-radius:7px;background:rgba(14,22,32,.72);padding:18px}.analysis-summary h2,.simulation-layer h2{margin:5px 0 0;font-size:22px}.eyebrow{color:#68798b;font-size:9px;font-weight:700;letter-spacing:.11em;text-transform:uppercase}.analysis-summary>p{color:#aab7c4;font-size:13px;line-height:1.6}.trajectory{position:relative;height:140px;margin-top:20px}.trajectory-line{position:absolute;right:0;left:0;height:2px;background:rgba(255,255,255,.07)}.trajectory-line.safe{top:54px;border-top:1px dashed #56697a;background:transparent}.trajectory-line.safe::before{content:"85°C safe limit";position:absolute;top:-18px;color:#68798a;font-size:9px}.trajectory svg{width:100%;height:100%}.cause-panel{display:flex;flex-direction:column;justify-content:center}.cause-row{margin-top:18px}.cause-row div:first-child{display:flex;justify-content:space-between;color:#aab7c4;font-size:11px}.cause-bar{height:5px;margin-top:7px;border-radius:5px;background:rgba(255,255,255,.07);overflow:hidden}.cause-bar span{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#28bad6,#f05288)}
        .simulation-layer{grid-template-columns:1fr auto 1fr;align-items:center;gap:18px}.simulation-arrow{color:#55d7ef}.simulation-arrow svg{width:24px}.sim-temp{margin-top:22px;font-size:52px;font-weight:750}.simulation-before .sim-temp{color:#f4779f}.simulation-after .sim-temp{color:#69dca2}.sim-state{display:inline-flex;margin-top:12px;border:1px solid rgba(255,255,255,.09);border-radius:99px;padding:5px 9px;color:#9eacba;font-size:10px}.simulation-after{border-color:rgba(69,197,138,.2);background:rgba(69,197,138,.04)}.run-simulation{display:inline-flex;min-height:42px;align-items:center;gap:7px;margin-top:22px;border:1px solid #20bedb;border-radius:6px;background:#1198b2;padding:0 16px;color:#041419;font-size:11px;font-weight:700}.run-simulation svg{width:14px}
        .rg-statusbar{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:10px;border:1px solid rgba(255,255,255,.08);border-radius:7px;background:rgba(12,19,28,.76);padding:11px 14px}.rg-stat{display:flex;align-items:center;gap:8px}.rg-stat svg{width:14px;color:#627385}.rg-stat span,.rg-stat strong{display:block}.rg-stat span{color:#68798a;font-size:8px;text-transform:uppercase}.rg-stat strong{margin-top:2px;color:#dce7ee;font-size:12px}.rg-stat.accent strong{color:var(--accent)}
        .rg-ai{display:flex;min-width:0;flex-direction:column;border-left:1px solid rgba(255,255,255,.09);background:rgba(8,13,20,.96);padding:18px}.ai-head{display:flex;align-items:center;gap:10px;border-bottom:1px solid rgba(255,255,255,.08);padding-bottom:14px}.ai-orb{position:relative;display:grid;width:42px;height:42px;place-items:center;border:1px solid rgba(72,215,239,.25);border-radius:50%}.ai-orb::before{content:"";position:absolute;inset:7px;border:1px solid rgba(72,215,239,.3);border-radius:50%;animation:orb 3s ease-in-out infinite}.ai-orb span{width:13px;height:13px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#effeff,#45d7ef 40%,#08788f 80%);box-shadow:0 0 17px rgba(72,215,239,.65)}.orb-safe span{background:radial-gradient(circle at 35% 30%,#ecfdf5,#6ee7b7 40%,#047857 80%)}.ai-name{display:flex;align-items:center;gap:6px}.ai-name h2{margin:0;font-size:15px}.ai-name span{border:1px solid rgba(69,197,138,.24);border-radius:99px;padding:2px 5px;color:#72d5a1;font-size:8px;font-weight:700;text-transform:uppercase}.ai-head p{margin:3px 0 0;color:#718095;font-size:9px}.ai-head button{display:grid;width:38px;height:38px;margin-left:auto;place-items:center;border:1px solid rgba(255,255,255,.08);border-radius:6px;background:rgba(255,255,255,.025);color:#8090a0}.ai-head button svg{width:15px}.ai-sync{display:flex;align-items:center;gap:7px;margin-top:12px;border:1px solid rgba(72,215,239,.11);border-radius:6px;background:rgba(72,215,239,.035);padding:8px 9px;color:#8ca1b2;font-size:9px}.ai-sync svg{width:13px;color:#6bd9ed}.ai-sync b{margin-left:auto;color:#617386}
        .ai-body{flex:1;min-height:0;padding:17px 0 12px}.ai-bubble{display:flex;gap:8px}.bot-mark{display:grid;width:28px;height:28px;place-items:center;border:1px solid rgba(72,215,239,.2);border-radius:6px;color:#70dcef}.bot-mark svg{width:14px}.ai-bubble>div:last-child{flex:1;border:1px solid rgba(255,255,255,.08);border-radius:4px 8px 8px 8px;background:#101923;padding:12px}.ai-bubble span,.ai-cause span,.ai-action-card span{color:#6c7c8d;font-size:8px;font-weight:700;letter-spacing:.09em;text-transform:uppercase}.ai-bubble p{margin:6px 0 0;color:#c6d1da;font-size:13px;line-height:1.58}.ai-cause{margin:12px 0 0 36px;border-left:2px solid rgba(72,215,239,.26);padding:2px 0 2px 11px}.ai-cause span{display:flex;align-items:center;gap:5px}.ai-cause svg{width:12px;color:#68d9ed}.ai-cause strong{display:block;margin-top:7px;color:#dce7ee;font-size:12px}.ai-cause p{margin:3px 0 0;color:#8596a7;font-size:10px}.ai-action-card{margin:15px 0 0 36px;border:1px solid rgba(72,215,239,.16);border-radius:7px;background:rgba(72,215,239,.045);padding:12px}.ai-action-card span{display:flex;align-items:center;gap:6px}.ai-action-card svg{width:13px;color:#6edcef}.ai-action-card strong{display:block;margin-top:8px;color:#e5f7fa;font-size:13px;line-height:1.4}.ai-action-card p{margin:6px 0 0;color:#7f9ca6;font-size:10px}.action-safe{border-color:rgba(69,197,138,.2);background:rgba(69,197,138,.05)}.action-safe strong,.action-safe svg{color:#76d9a5}
        .ai-reply{margin:10px 0 0 36px;border-left:2px solid rgba(72,215,239,.3);padding:4px 10px}.ai-reply span{color:#6bd9ed;font-size:8px;font-weight:700;text-transform:uppercase}.ai-reply p{margin:4px 0 0;color:#afbdc8;font-size:11px;line-height:1.45}.ai-buttons{display:grid;grid-template-columns:.75fr 1.25fr;gap:7px;border-top:1px solid rgba(255,255,255,.08);padding-top:12px}.ai-buttons button{display:inline-flex;min-height:42px;align-items:center;justify-content:center;gap:7px;border-radius:6px;font-size:10px;font-weight:700}.ai-buttons svg{width:14px}.simulate{border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.035);color:#c4cfd8}.accept{border:1px solid #22bfdc;background:#1397b1;color:#041318}.ai-buttons button:disabled{cursor:not-allowed;opacity:.42}.ai-input{display:grid;grid-template-columns:minmax(0,1fr) 38px 38px;gap:5px;margin-top:8px}.ai-input input{min-width:0;height:40px;border:1px solid rgba(255,255,255,.09);border-radius:6px;background:#0d151e;padding:0 10px;color:white;font-size:11px}.ai-input button{display:grid;width:38px;height:40px;place-items:center;border:1px solid rgba(255,255,255,.09);border-radius:6px;background:rgba(255,255,255,.03);color:#7790a1}.ai-input svg{width:14px}.ai-input button:disabled{cursor:not-allowed;opacity:.35}.ai-note{margin:7px 0 0;color:#566677;font-size:8px;text-align:center}
        @keyframes thermal-pulse{50%{box-shadow:0 0 0 1px rgba(240,82,136,.2),0 0 30px rgba(240,82,136,.32),inset 0 0 34px rgba(0,0,0,.36)}}@keyframes orb{50%{transform:scale(1.12);opacity:.45}}
        @media(max-height:760px) and (min-width:761px){
          .rg-ai{padding:14px 16px}.ai-head{padding-bottom:10px}.ai-orb{width:38px;height:38px}
          .ai-sync{margin-top:9px;padding:7px 9px}.ai-body{padding:11px 0 8px}
          .ai-bubble>div:last-child{padding:10px}.ai-bubble p{font-size:11px;line-height:1.45}
          .ai-cause{margin-top:8px}.ai-cause strong{margin-top:4px}.ai-action-card{margin-top:10px;padding:10px}
          .ai-action-card strong{margin-top:5px;font-size:11px}.ai-action-card p{margin-top:3px}
          .ai-buttons{padding-top:9px}.ai-buttons button{min-height:38px}.ai-input input,.ai-input button{height:36px}
          .ai-note{margin-top:4px}
        }
        @media(max-width:1000px){.rg-shell{grid-template-columns:1fr 330px}.deck-rack{width:160px;height:280px;transform:translate(-50%,-50%) translateX(calc(var(--deck-offset) * 166px + var(--drag-x, 0px))) scale(calc(1 - var(--deck-distance) * .1))}.analysis-layer{grid-template-columns:1fr}}
        @media(max-width:760px){html,body{overflow:auto}.rg-shell{display:block;height:auto;min-height:100vh}.rg-ai{border-top:1px solid rgba(255,255,255,.09);border-left:0}.overview-layer,.analysis-layer,.simulation-layer{min-height:430px}.rg-statusbar{grid-template-columns:1fr 1fr}.layer-button kbd{display:none}}
        @media(prefers-reduced-motion:reduce){.tower-critical[aria-pressed=true],.ai-orb::before{animation:none}.deck-rack{transition:none}}
      `}</style>
    </section>
  );
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

function LayerButton({ active, onClick, icon: Icon, label, shortcut }: { active: boolean; onClick: () => void; icon: typeof Layers3; label: string; shortcut: string }) {
  return <button className={`layer-button ${active ? "active" : ""}`} type="button" onClick={onClick}><Icon />{label}<kbd>{shortcut}</kbd></button>;
}

function OverviewLayer({ activeIndex, accepted, onMove, onSelect, onWheel }: {
  activeIndex: number; accepted: boolean; onMove: (direction: number) => void; onSelect: (index: number) => void;
  onWheel: (event: WheelEvent) => void;
}) {
  const [dragOffset, setDragOffset] = useState(0);
  const dragStart = useRef<number | null>(null);
  const dragTime = useRef(0);
  const didDrag = useRef(false);

  function startDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if ((event.target as HTMLElement).closest(".deck-controls")) return;
    dragStart.current = event.clientX;
    dragTime.current = performance.now();
    didDrag.current = false;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function updateDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if (dragStart.current === null) return;
    const rawOffset = event.clientX - dragStart.current;
    const atStart = activeIndex === 0 && rawOffset > 0;
    const atEnd = activeIndex === racks.length - 1 && rawOffset < 0;
    const offset = atStart || atEnd ? rawOffset * .22 : rawOffset;
    didDrag.current = Math.abs(rawOffset) > 5;
    setDragOffset(Math.max(-240, Math.min(240, offset)));
  }

  function finishDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if (dragStart.current === null) return;
    const distance = event.clientX - dragStart.current;
    const elapsed = Math.max(1, performance.now() - dragTime.current);
    const velocity = Math.abs(distance) / elapsed;
    if (Math.abs(distance) > 72 || (Math.abs(distance) > 24 && velocity > .45)) {
      onMove(distance > 0 ? -1 : 1);
    }
    dragStart.current = null;
    setDragOffset(0);
  }

  function cancelDrag() {
    dragStart.current = null;
    setDragOffset(0);
  }

  function selectRack(index: number) {
    if (!didDrag.current) onSelect(index);
    didDrag.current = false;
  }

  return (
    <div
      className={`overview-layer ${dragStart.current !== null ? "dragging" : ""}`}
      style={{ "--drag-x": `${dragOffset}px` } as CSSProperties}
      onWheel={onWheel}
      onPointerDown={startDrag}
      onPointerMove={updateDrag}
      onPointerUp={finishDrag}
      onPointerCancel={cancelDrag}
    >
      <div className="deck-caption"><span>Priority carousel</span><strong>Swipe, scroll or use arrow keys</strong></div>
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
          return <DeckRack key={rack.id} rack={rack} rank={index + 1} state={state} active={index === activeIndex} style={style} onClick={() => selectRack(index)} />;
        })}
      </div>
    </div>
  );
}

function DeckRack({ rack, rank, state, active, style, onClick }: { rack: Rack; rank: number; state: RackState; active: boolean; style: CSSProperties; onClick: () => void }) {
  const Icon = state === "critical" ? CircleAlert : state === "warning" ? TriangleAlert : CheckCircle2;
  return (
    <div className="deck-rack" style={style}>
      <button className={`tower tower-${state}`} aria-pressed={active} onClick={onClick}>
        <span className="tower-rank">Priority {String(rank).padStart(2, "0")}</span>
        <div className="tower-head"><Server /><strong>{rack.id}</strong><i><Icon /></i></div>
        <div className="tower-temp">{rack.temperature}<small>°C</small></div>
        <div className="tower-slots">{Array.from({ length: 7 }).map((_, i) => <span key={i} />)}</div>
        <div className="tower-load"><div className="tower-track"><span style={{ width: `${rack.load}%` }} /></div><b>{rack.load}%</b></div>
        <div className="tower-foot"><span>Equilibrium</span><strong>{rack.predicted}°C</strong></div>
      </button>
    </div>
  );
}

function AnalysisLayer({ rack }: { rack: Rack }) {
  return (
    <div className="analysis-layer">
      <div className="analysis-summary">
        <span className="eyebrow">AI thermal trajectory</span><h2>{rack.id} converges toward {rack.predicted}°C</h2>
        <p>{rack.cause}. {rack.secondaryCause}. Guardian AI expects the rack to {rack.state === "critical" ? "cross the safe limit without intervention" : "remain within its safe operating envelope"}.</p>
        <div className="trajectory"><div className="trajectory-line safe" /><svg viewBox="0 0 500 140"><path d={rack.state === "critical" ? "M5 120 C120 105 210 75 300 52 S420 22 495 15" : "M5 110 C130 92 220 78 310 72 S430 68 495 66"} fill="none" stroke={rack.state === "critical" ? "#f05288" : "#48d7ef"} strokeWidth="4" strokeLinecap="round" /></svg></div>
      </div>
      <div className="cause-panel"><span className="eyebrow">Contributing factors</span><CauseRow label={rack.cause} value={68} /><CauseRow label={rack.secondaryCause} value={24} /><CauseRow label="Ambient temperature" value={8} /></div>
    </div>
  );
}

function CauseRow({ label, value }: { label: string; value: number }) {
  return <div className="cause-row"><div><span>{label}</span><b>{value}%</b></div><div className="cause-bar"><span style={{ width: `${value}%` }} /></div></div>;
}

function SimulationLayer({ rack, outcome, simulated, accepted, onRun }: { rack: Rack; outcome: number; simulated: boolean; accepted: boolean; onRun: () => void }) {
  return (
    <div className="simulation-layer">
      <div className="simulation-before"><span className="eyebrow">Without action</span><h2>{rack.id}</h2><div className="sim-temp">{rack.predicted}°C</div><span className="sim-state">{rack.state === "critical" ? "Slowdown likely" : "Monitor"}</span></div>
      <div className="simulation-arrow"><ArrowRight /></div>
      <div className="simulation-after"><span className="eyebrow">Recommended outcome</span><h2>{rack.recommendation}</h2><div className="sim-temp">{simulated || accepted ? `${outcome}°C` : "—"}</div><span className="sim-state">{simulated || accepted ? "Slowdown avoided" : "Ready to simulate"}</span>{!simulated && !accepted && <button className="run-simulation" onClick={onRun}><Play />Run simulation</button>}</div>
    </div>
  );
}

function Stat({ icon: Icon, label, value, accent = false }: { icon: typeof Thermometer; label: string; value: string; accent?: boolean }) {
  return <div className={`rg-stat ${accent ? "accent" : ""}`}><Icon /><div><span>{label}</span><strong>{value}</strong></div></div>;
}
