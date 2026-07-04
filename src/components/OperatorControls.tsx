type Props = {
  accepted: boolean;
  isSpeaking: boolean;
  speechSupported: boolean;
  technicalOpen: boolean;
  onAccept: () => void;
  onAskWhy: () => void;
  onShowAlternatives: () => void;
  onToggleTechnical: () => void;
  onSpeak: () => void;
};

export function OperatorControls(props: Props) {
  return (
    <div className="mt-4">
      <div className="flex flex-col gap-3 sm:flex-row">
        <button className="button-primary" onClick={props.onAccept} disabled={props.accepted}>
          {props.accepted ? "Recommendation accepted" : "Accept recommendation"}
        </button>
        <button className="button-secondary" onClick={props.onAskWhy}>Ask why</button>
        <button className="button-secondary" onClick={props.onShowAlternatives}>Show alternatives</button>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          className="button-quiet"
          onClick={props.onSpeak}
          disabled={!props.speechSupported || props.isSpeaking}
          title={!props.speechSupported ? "Speech is unavailable in this browser" : "Read the current message aloud"}
        >
          <span className="speaker-icon" aria-hidden="true" />
          {props.isSpeaking ? "Speaking…" : "Speak alert"}
        </button>
        <button
          className="mic-button"
          type="button"
          aria-label="Voice commands coming soon"
          title="Voice commands coming soon"
          disabled
        >
          <span className="mic-icon" aria-hidden="true" />
        </button>
        <button className="button-text ml-auto" onClick={props.onToggleTechnical}>
          {props.technicalOpen ? "Hide technical details" : "Technical details"}
          <span aria-hidden="true">{props.technicalOpen ? " ↑" : " ↓"}</span>
        </button>
      </div>
    </div>
  );
}
