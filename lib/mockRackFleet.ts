import type {
  GpuSnapshot,
  RackSnapshot,
} from "./rackGuardianTypes";

export type {
  GpuSnapshot,
  RackSnapshot,
} from "./rackGuardianTypes";

type GpuStatus = GpuSnapshot["status"];

type RackProfile = {
  rackId: string;
  temperatures: number[];
  predictions: number[];
  utilizations: number[];
  powers: number[];
  cause: string;
  secondaryCause: string;
  recommendation: string;
  slowdown: string;
};

function gpuStatus(predictedTemperature: number): GpuStatus {
  if (predictedTemperature >= 85) return "critical";
  if (predictedTemperature >= 78) return "warning";
  return "normal";
}

function buildRack(profile: RackProfile): RackSnapshot {
  const gpus: GpuSnapshot[] = profile.temperatures.map((temperature, index) => {
    const predictedTemperature = profile.predictions[index];
    const status = gpuStatus(predictedTemperature);
    const coolingGain = status === "critical" ? 12 : status === "warning" ? 7 : 3;
    const temperatureAfter = predictedTemperature - coolingGain;

    return {
      gpu_id: `gpu-${String(index + 1).padStart(2, "0")}`,
      rack_id: profile.rackId,
      status,
      temperature_c: temperature,
      predicted_temperature_c: predictedTemperature,
      utilization_percent: profile.utilizations[index],
      power_draw_w: profile.powers[index],
      risk_score: Math.min(0.98, Math.max(0.08, (predictedTemperature - 60) / 32)),
      risk_reason: status === "critical"
        ? "Sustained utilization is pushing the predicted temperature above the safe limit."
        : status === "warning"
          ? "Temperature is rising under load and should be checked before it reaches the safe limit."
          : "Temperature remains stable with sufficient cooling headroom.",
      recommended_action: status === "critical"
        ? "Increase ventilation by 20%"
        : status === "warning"
          ? "Rebalance the workload"
          : "Continue monitoring",
      simulated_solution: {
        action: status === "critical"
          ? "Increase ventilation by 20%"
          : status === "warning"
            ? "Rebalance the workload"
            : "Maintain current settings",
        temperature_after_c: temperatureAfter,
        cooling_gain_c: coolingGain,
        risk_after: gpuStatus(temperatureAfter),
        performance_impact: status === "normal" ? "none" : "low",
      },
    };
  });
  const criticalCount = gpus.filter((gpu) => gpu.status === "critical").length;
  const warningCount = gpus.filter((gpu) => gpu.status === "warning").length;
  const normalCount = gpus.length - criticalCount - warningCount;

  return {
    rack_id: profile.rackId,
    gpu_count: gpus.length,
    gpus_to_check: warningCount + criticalCount,
    normal_count: normalCount,
    warning_count: warningCount,
    critical_count: criticalCount,
    status: criticalCount > 0 ? "critical" : warningCount > 0 ? "warning" : "normal",
    gpus,
    cause: profile.cause,
    secondary_cause: profile.secondaryCause,
    recommendation: profile.recommendation,
    slowdown: profile.slowdown,
  };
}

const profiles: RackProfile[] = [
  {
    rackId: "rack-01",
    temperatures: [82, 76, 74, 71, 69, 68, 66, 64],
    predictions: [88, 81, 79, 75, 73, 72, 70, 68],
    utilizations: [96, 89, 84, 73, 68, 61, 56, 48],
    powers: [315, 296, 284, 252, 238, 224, 210, 192],
    cause: "High sustained GPU utilization",
    secondaryCause: "Cooling flow reduced",
    recommendation: "Inspect GPU-01 and rebalance its workload.",
    slowdown: "6 min 40 sec",
  },
  {
    rackId: "rack-02",
    temperatures: [74, 73, 71, 69, 68, 67, 65, 63],
    predictions: [81, 79, 76, 74, 73, 71, 69, 67],
    utilizations: [86, 82, 77, 70, 66, 60, 53, 46],
    powers: [286, 278, 265, 244, 235, 219, 204, 188],
    cause: "Two GPUs are warming under sustained load",
    secondaryCause: "Cooling capacity remains available",
    recommendation: "Monitor GPU-01 and GPU-02.",
    slowdown: "31 min",
  },
  {
    rackId: "rack-03",
    temperatures: [68, 67, 66, 65, 64, 63, 62, 61],
    predictions: [74, 73, 72, 71, 70, 69, 68, 67],
    utilizations: [72, 69, 65, 62, 58, 54, 49, 44],
    powers: [248, 239, 232, 224, 215, 207, 196, 184],
    cause: "Balanced workload distribution",
    secondaryCause: "Healthy cooling reserve",
    recommendation: "No action required.",
    slowdown: "None",
  },
  {
    rackId: "rack-04",
    temperatures: [84, 82, 77, 72, 70, 68, 66, 64],
    predictions: [91, 87, 82, 76, 74, 72, 70, 68],
    utilizations: [98, 94, 88, 75, 69, 62, 55, 47],
    powers: [322, 310, 298, 258, 241, 226, 208, 190],
    cause: "High power draw on two GPUs",
    secondaryCause: "Warm intake air",
    recommendation: "Inspect GPU-01 and GPU-02 first.",
    slowdown: "4 min 20 sec",
  },
];

export const rackFleet: RackSnapshot[] = profiles
  .map(buildRack)
  .sort(
    (a, b) =>
      b.critical_count - a.critical_count ||
      b.warning_count - a.warning_count,
  );
