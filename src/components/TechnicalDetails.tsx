type Props = {
  open: boolean;
  currentTemperature: number;
  predictedEquilibrium: number;
  timeBeforeThrottling: string;
  confidenceScore: number;
  accepted: boolean;
};

export function TechnicalDetails(props: Props) {
  if (!props.open) return null;
  const safe = props.accepted;
  return (
    <section className="technical-panel mt-5" aria-label="Technical details">
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
        <Detail label="Current" value={`${props.currentTemperature}°C`} />
        <Detail label="Equilibrium" value={`${props.predictedEquilibrium}°C`} />
        <Detail label="Before throttling" value={safe ? "Avoided" : props.timeBeforeThrottling} />
        <Detail label="Confidence" value={`${props.confidenceScore}%`} />
      </div>
      <div className="mt-6 grid gap-6 sm:grid-cols-[1.4fr_0.6fr]">
        <div>
          <p className="eyebrow">Temperature projection</p>
          <svg className="mt-3 h-28 w-full" viewBox="0 0 520 120" role="img" aria-label="Temperature projection curve">
            <line x1="10" y1="47" x2="510" y2="47" stroke="#475569" strokeDasharray="5 7" />
            <text x="12" y="39" fill="#64748b" fontSize="11">85°C safe limit</text>
            <path d={safe ? "M10 88 C130 55 180 58 260 74 S410 90 510 94" : "M10 88 C130 72 190 58 270 44 S420 25 510 20"} fill="none" stroke={safe ? "#34d399" : "#f87171"} strokeWidth="3" strokeLinecap="round" />
            <circle cx="510" cy={safe ? "94" : "20"} r="5" fill={safe ? "#34d399" : "#f87171"} />
          </svg>
        </div>
        <div>
          <p className="eyebrow">Main causes</p>
          <Cause label="GPU power draw" value={68} />
          <Cause label="Reduced cooling flow" value={24} />
          <Cause label="Ambient rise" value={8} />
        </div>
      </div>
    </section>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div><p className="text-xs text-slate-500">{label}</p><p className="mt-1 font-semibold text-slate-200">{value}</p></div>;
}

function Cause({ label, value }: { label: string; value: number }) {
  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-slate-400"><span>{label}</span><span>{value}%</span></div>
      <div className="mt-1 h-1 rounded-full bg-white/10"><div className="h-full rounded-full bg-cyan-400/60" style={{ width: `${value}%` }} /></div>
    </div>
  );
}
