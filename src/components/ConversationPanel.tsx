export function ConversationPanel({
  message,
  complete,
  safe
}: {
  message: string;
  complete: boolean;
  safe: boolean;
}) {
  return (
    <section className="conversation-panel" aria-live="polite">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${safe ? "bg-emerald-400" : "bg-cyan-300"}`} />
        <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-400">
          Co-pilot assessment
        </p>
        </div>
        <span className={`decision-label ${safe ? "decision-safe" : "decision-urgent"}`}>
          {safe ? "Resolved" : "Action needed"}
        </span>
      </div>
      <p className="min-h-[6.75rem] text-[1.35rem] font-medium leading-[1.48] text-white sm:min-h-[5.9rem] sm:text-[1.65rem]">
        {message}
        {!complete && <span className="typing-cursor" aria-hidden="true" />}
      </p>
      <p className="mt-3 text-sm text-slate-400">
        {safe
          ? "The thermal trajectory is back inside the safe operating envelope."
          : "Expected result: 90°C → 74°C. Slowdown avoided. You approve every action."}
      </p>
    </section>
  );
}
