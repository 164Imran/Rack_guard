type Option = {
  name: string;
  outcome: string;
  consequence: string;
  tone: "critical" | "safe";
};

type OptionsModalProps = {
  open: boolean;
  options: readonly Option[];
  onClose: () => void;
};

export function OptionsModal({ open, options, onClose }: OptionsModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-20 flex items-end justify-center bg-slate-950/80 px-4 py-4 backdrop-blur-sm sm:items-center"
      role="dialog"
      aria-modal="true"
      aria-labelledby="options-title"
      onMouseDown={onClose}
    >
      <section
        className="w-full max-w-lg rounded-lg border border-white/10 bg-[#111923] p-5 shadow-calm"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="options-title" className="text-xl font-semibold text-white">
              Other options
            </h2>
            <p className="mt-1 text-sm text-slate-400">Choose the response that best fits current operations.</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="h-9 w-9 rounded-md border border-white/10 text-lg text-slate-300 hover:bg-white/[0.06] focus:outline-none focus:ring-2 focus:ring-cyan-300"
            aria-label="Close options"
          >
            ×
          </button>
        </div>

        <div className="mt-5 space-y-3">
          {options.map((option) => (
            <article key={option.name} className="rounded-md border border-white/10 bg-white/[0.03] p-4">
              <div className="flex items-center justify-between gap-4">
                <h3 className="font-semibold text-white">{option.name}</h3>
                <span
                  className={`rounded-full px-3 py-1 text-sm font-semibold ${
                    option.tone === "safe"
                      ? "bg-emerald-400/10 text-emerald-200"
                      : "bg-red-500/10 text-red-200"
                  }`}
                >
                  {option.outcome}
                </span>
              </div>
              <p
                className={`mt-2 text-sm ${
                  option.tone === "safe" ? "text-emerald-200" : "text-red-200"
                }`}
              >
                {option.consequence}
              </p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
