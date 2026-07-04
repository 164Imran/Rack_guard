type OutcomeStateProps = {
  message: string;
};

export function OutcomeState({ message }: OutcomeStateProps) {
  return (
    <section className="rounded-lg border border-emerald-400/25 bg-emerald-400/10 px-5 py-4">
      <p className="text-lg font-semibold text-emerald-100">{message}</p>
      <p className="mt-1 text-sm text-emerald-200">Slowdown avoided</p>
    </section>
  );
}
