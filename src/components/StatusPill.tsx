export function StatusPill({ safe, label }: { safe: boolean; label: string }) {
  return (
    <span className={`status-pill ${safe ? "status-safe" : "status-critical"}`}>
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}
