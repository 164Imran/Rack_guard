export function AssistantOrb({ isSpeaking, safe }: { isSpeaking: boolean; safe: boolean }) {
  return (
    <div className="assistant-identity">
      <div
        className={`assistant-orb ${isSpeaking ? "is-speaking" : ""} ${safe ? "is-safe" : ""}`}
        aria-label={isSpeaking ? "Assistant is speaking" : "Assistant is monitoring"}
      >
        <span className="orb-core" />
        <span className="orb-ring orb-ring-one" />
        <span className="orb-ring orb-ring-two" />
      </div>
      <div className="assistant-presence">
        <span className="monitoring-dot" />
        <span>Monitoring cluster</span>
      </div>
    </div>
  );
}
