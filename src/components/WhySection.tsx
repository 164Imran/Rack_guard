type WhySectionProps = {
  reasons: readonly string[];
};

export function WhySection({ reasons }: WhySectionProps) {
  return (
    <section className="rounded-lg border border-white/10 bg-slate-950/62 p-5">
      <h2 className="text-xl font-semibold text-white">Why this is happening</h2>
      <ul className="mt-4 space-y-3">
        {reasons.map((reason) => (
          <li key={reason} className="flex gap-3 text-base text-slate-300">
            <span className="mt-2 h-2 w-2 shrink-0 rounded-full bg-blue-300" aria-hidden="true" />
            <span>{reason}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
