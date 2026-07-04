type HeaderProps = {
  productName: string;
  status: string;
  badge: string;
  accepted: boolean;
};

export function Header({ productName, status, badge, accepted }: HeaderProps) {
  return (
    <header className="flex flex-col gap-3 rounded-lg border border-white/10 bg-slate-950/62 px-5 py-4 shadow-calm backdrop-blur sm:flex-row sm:items-center sm:justify-between">
      <div>
        <p className="text-sm font-medium text-blue-200">AI thermal co-pilot</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-normal text-white sm:text-3xl">
          {productName}
        </h1>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded-full border px-3 py-1 text-sm font-medium ${
            accepted
              ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-200"
              : "border-red-400/35 bg-red-500/10 text-red-200"
          }`}
        >
          {status}
        </span>
        <span className="rounded-full border border-blue-300/25 bg-blue-400/10 px-3 py-1 text-sm font-medium text-blue-100">
          {badge}
        </span>
      </div>
    </header>
  );
}
