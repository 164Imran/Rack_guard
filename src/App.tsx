import { useEffect, useMemo, useState } from "react";
import { AssistantOrb } from "./components/AssistantOrb";
import { ConversationPanel } from "./components/ConversationPanel";
import { AlertSummaryCard } from "./components/AlertSummaryCard";
import { RecommendedActionCard } from "./components/RecommendedActionCard";
import { OperatorControls } from "./components/OperatorControls";
import { OptionsModal } from "./components/OptionsModal";
import { TechnicalDetails } from "./components/TechnicalDetails";
import { StatusPill } from "./components/StatusPill";
import { rackGuardianMock } from "./data/rackGuardianMock";

export default function App() {
  const [accepted, setAccepted] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [technicalOpen, setTechnicalOpen] = useState(false);
  const [assistantMessage, setAssistantMessage] = useState<string>(rackGuardianMock.messages.alert);
  const [visibleMessage, setVisibleMessage] = useState("");
  const [speechEnabled, setSpeechEnabled] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const speechSupported = useMemo(
    () => typeof window !== "undefined" && "speechSynthesis" in window,
    []
  );

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setVisibleMessage(assistantMessage);
      return;
    }
    setVisibleMessage("");
    let index = 0;
    const timer = window.setInterval(() => {
      index += 1;
      setVisibleMessage(assistantMessage.slice(0, index));
      if (index >= assistantMessage.length) window.clearInterval(timer);
    }, 19);
    return () => window.clearInterval(timer);
  }, [assistantMessage]);

  useEffect(() => {
    return () => window.speechSynthesis?.cancel();
  }, []);

  function speak(text: string) {
    if (!speechSupported) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.94;
    utterance.pitch = 0.96;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }

  function handleSpeak() {
    setSpeechEnabled(true);
    speak(assistantMessage);
  }

  function handleAccept() {
    setAccepted(true);
    setOptionsOpen(false);
    setAssistantMessage(rackGuardianMock.messages.accepted);
    if (speechEnabled) speak(rackGuardianMock.messages.accepted);
  }

  function handleAskWhy() {
    setAssistantMessage(rackGuardianMock.messages.why);
    if (speechEnabled) speak(rackGuardianMock.messages.why);
  }

  const predictedTemperature = accepted ? 74 : 90;

  return (
    <main className="min-h-screen px-4 py-4 text-slate-100 sm:px-6 sm:py-5">
      <div className="mx-auto w-full max-w-[980px]">
        <header className="flex items-center justify-between border-b border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <div className="brand-mark" aria-hidden="true"><span /></div>
            <div>
              <p className="text-base font-semibold text-white">Rack Guardian</p>
              <p className="text-xs text-slate-500">Thermal co-pilot</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-xs text-slate-500 sm:inline">Live simulation</span>
            <StatusPill safe={accepted} label={accepted ? "Safe" : "Critical"} />
          </div>
        </header>

        <section className="mx-auto max-w-[860px] py-6 sm:py-8">
          <div className={`assistant-stage ${accepted ? "assistant-stage-safe" : ""}`}>
            <AssistantOrb isSpeaking={isSpeaking} safe={accepted} />
            <ConversationPanel
              message={visibleMessage}
              complete={visibleMessage.length === assistantMessage.length}
              safe={accepted}
            />
          </div>

          <OperatorControls
            accepted={accepted}
            isSpeaking={isSpeaking}
            speechSupported={speechSupported}
            technicalOpen={technicalOpen}
            onAccept={handleAccept}
            onAskWhy={handleAskWhy}
            onShowAlternatives={() => setOptionsOpen(true)}
            onToggleTechnical={() => setTechnicalOpen((open) => !open)}
            onSpeak={handleSpeak}
          />

          <div className="mt-4 grid gap-3 md:grid-cols-[0.78fr_1.22fr]">
            <AlertSummaryCard
              accepted={accepted}
              predictedTemperature={predictedTemperature}
            />
            <RecommendedActionCard accepted={accepted} />
          </div>

          <TechnicalDetails
            open={technicalOpen}
            currentTemperature={84}
            predictedEquilibrium={predictedTemperature}
            timeBeforeThrottling="6 min 40 sec"
            confidenceScore={92}
            accepted={accepted}
          />
        </section>
      </div>

      <OptionsModal
        open={optionsOpen}
        options={rackGuardianMock.alternatives}
        onClose={() => setOptionsOpen(false)}
      />
    </main>
  );
}
