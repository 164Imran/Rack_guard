export const rackGuardianMock = {
  messages: {
    alert:
      "Rack R-04 is converging toward an unsafe temperature. Slowdown expected in 6 minutes. I recommend moving the workload to Rack R-06.",
    why:
      "The current workload is drawing unusually high power for several minutes. That extra power becomes heat. Because cooling flow is also reduced, the rack cannot return to a safe thermal equilibrium.",
    accepted:
      "Action accepted. Workload migration simulated. Rack R-04 is now converging toward 74°C. Slowdown avoided."
  },
  alternatives: [
    { name: "No action", outcome: "90°C", consequence: "Slowdown likely", tone: "critical" },
    { name: "Increase cooling", outcome: "79°C", consequence: "Safe", tone: "safe" },
    { name: "Reduce GPU frequency", outcome: "76°C", consequence: "Safe", tone: "safe" }
  ]
} as const;
