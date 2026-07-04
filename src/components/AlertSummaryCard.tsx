import { StatusPill } from "./StatusPill";

export function AlertSummaryCard({
  accepted,
  predictedTemperature
}: {
  accepted: boolean;
  predictedTemperature: number;
}) {
  return (
    <section className="panel p-4 sm:p-5" aria-labelledby="alert-title">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Active rack</p>
          <h2 id="alert-title" className="mt-1 text-lg font-semibold text-white">Rack R-04</h2>
        </div>
        <StatusPill safe={accepted} label={accepted ? "Safe" : "Critical"} />
      </div>
      <div className="mt-4 flex items-end justify-between border-b border-white/10 pb-3">
        <div>
          <p className="text-sm text-slate-400">Predicted equilibrium</p>
          <p className={`mt-1 text-3xl font-semibold ${accepted ? "text-emerald-300" : "text-red-300"}`}>
            {predictedTemperature}°C
          </p>
        </div>
        <p className="pb-1 text-right text-sm text-slate-400">
          Current <strong className="block text-lg text-white">84°C</strong>
        </p>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
        <div><p className="text-slate-500">Safe limit</p><p className="mt-1 font-medium text-slate-200">85°C</p></div>
        <div><p className="text-slate-500">Slowdown</p><p className="mt-1 font-medium text-slate-200">{accepted ? "Avoided" : "6 min 40 sec"}</p></div>
      </div>
    </section>
  );
}
