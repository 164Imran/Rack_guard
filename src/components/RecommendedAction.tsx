type RecommendedActionProps = {
  action: string;
  expectedResult: string;
  accepted: boolean;
  onAccept: () => void;
  onShowOptions: () => void;
  onToggleTechnical: () => void;
  technicalOpen: boolean;
};

export function RecommendedAction({
  action,
  expectedResult,
  accepted,
  onAccept,
  onShowOptions,
  onToggleTechnical,
  technicalOpen
}: RecommendedActionProps) {
  return (
    <section className="rounded-lg border border-blue-300/20 bg-blue-400/[0.07] p-5">
      <p className="text-sm font-semibold text-blue-200">AI recommendation</p>
      <h2 className="mt-2 text-xl font-semibold text-white">Recommended action</h2>

      <div className="mt-4 rounded-lg border border-white/10 bg-slate-950/58 p-4">
        <p className="text-lg font-semibold leading-7 text-white">{action}</p>
        <p className="mt-3 text-base leading-7 text-slate-300">{expectedResult}</p>
      </div>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
        <button
          type="button"
          onClick={onAccept}
          disabled={accepted}
          className="inline-flex min-h-12 items-center justify-center rounded-md bg-blue-500 px-5 py-3 text-base font-semibold text-white transition hover:bg-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200 focus:ring-offset-2 focus:ring-offset-slate-950 disabled:cursor-not-allowed disabled:bg-emerald-600 disabled:text-emerald-50"
        >
          {accepted ? "Recommendation accepted" : "Accept recommendation"}
        </button>
        <button
          type="button"
          onClick={onShowOptions}
          className="inline-flex min-h-12 items-center justify-center rounded-md border border-white/15 bg-white/[0.04] px-5 py-3 text-base font-semibold text-slate-100 transition hover:bg-white/[0.08] focus:outline-none focus:ring-2 focus:ring-blue-200 focus:ring-offset-2 focus:ring-offset-slate-950"
        >
          See other options
        </button>
        <button
          type="button"
          onClick={onToggleTechnical}
          className="inline-flex min-h-12 items-center justify-center rounded-md px-2 py-3 text-base font-semibold text-blue-200 transition hover:text-blue-100 focus:outline-none focus:ring-2 focus:ring-blue-200 focus:ring-offset-2 focus:ring-offset-slate-950"
        >
          {technicalOpen ? "Hide technical details" : "Show technical details"}
        </button>
      </div>
    </section>
  );
}
