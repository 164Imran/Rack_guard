export function RecommendedActionCard({ accepted }: { accepted: boolean }) {
  return (
    <section className={`panel p-4 sm:p-5 ${accepted ? "panel-success" : "panel-action"}`}>
      <p className="eyebrow">{accepted ? "Action completed" : "Recommended action"}</p>
      <h2 className="mt-2 text-lg font-semibold leading-snug text-white">
        {accepted
          ? "Workload migration simulated"
          : "Move workload llm-ft-2841 from Rack R-04 to Rack R-06."}
      </h2>
      <p className="mt-3 text-[15px] leading-relaxed text-slate-300">
        {accepted
          ? "Rack R-04 is returning to a safe thermal equilibrium."
          : "Predicted temperature drops from 90°C to 74°C. Thermal slowdown avoided."}
      </p>
      <div className={`mt-3 flex items-center gap-2 text-sm font-semibold ${accepted ? "text-emerald-300" : "text-cyan-200"}`}>
        <span aria-hidden="true">{accepted ? "✓" : "→"}</span>
        {accepted ? "Slowdown avoided" : "Recommendation only — you remain in control"}
      </div>
    </section>
  );
}
