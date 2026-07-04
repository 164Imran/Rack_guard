type MainAlertCardProps = {
  title: string;
  subtitle: string;
  predictedEquilibrium: number;
  safeLimit: number;
  timeBeforeSlowdown: string;
  riskLevel: string;
  accepted: boolean;
};

export function MainAlertCard({
  title,
  subtitle,
  predictedEquilibrium,
  safeLimit,
  timeBeforeSlowdown,
  riskLevel,
  accepted
}: MainAlertCardProps) {
  return (
    <article className="rounded-lg border border-white/10 bg-slate-950/72 p-6 shadow-calm sm:p-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className={`text-sm font-semibold ${accepted ? "text-emerald-200" : "text-red-200"}`}>
            {accepted ? "Thermal risk resolved" : "Immediate attention needed"}
          </p>
          <h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-normal text-white sm:text-4xl">
            {accepted ? "Rack R-04 is now safe" : title}
          </h2>
          <p className="mt-3 max-w-xl text-lg leading-7 text-slate-300">
            {accepted ? "The simulated migration brings the rack back below the safe limit." : subtitle}
          </p>
        </div>

        <div
          className={`rounded-lg border px-4 py-3 text-center ${
            accepted
              ? "border-emerald-400/30 bg-emerald-400/10"
              : "border-red-400/30 bg-red-500/10"
          }`}
        >
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-300">Risk level</p>
          <p className={`mt-1 text-2xl font-semibold ${accepted ? "text-emerald-200" : "text-red-200"}`}>
            {riskLevel}
          </p>
        </div>
      </div>

      <dl className="mt-8 grid gap-3 sm:grid-cols-3">
        <Metric
          label="Predicted equilibrium"
          value={`${predictedEquilibrium}°C`}
          tone={accepted ? "safe" : "critical"}
        />
        <Metric label="Safe limit" value={`${safeLimit}°C`} tone="neutral" />
        <Metric
          label={accepted ? "Outcome" : "Time before slowdown"}
          value={accepted ? "Slowdown avoided" : timeBeforeSlowdown}
          tone={accepted ? "safe" : "warning"}
        />
      </dl>
    </article>
  );
}

function Metric({
  label,
  value,
  tone
}: {
  label: string;
  value: string;
  tone: "critical" | "warning" | "safe" | "neutral";
}) {
  const valueColor = {
    critical: "text-red-200",
    warning: "text-amber-200",
    safe: "text-emerald-200",
    neutral: "text-slate-100"
  }[tone];

  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
      <dt className="text-sm text-slate-400">{label}</dt>
      <dd className={`mt-2 text-2xl font-semibold ${valueColor}`}>{value}</dd>
    </div>
  );
}
