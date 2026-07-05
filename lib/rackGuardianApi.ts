import { rackFleet } from "./mockRackFleet";
import type {
  ActionScore,
  ActionSimulation,
  ActionType,
  BackendHealth,
  CurrentGpu,
  CurrentGpuResponse,
  FinalRecommendation,
  GpuDecisionData,
  GpuSnapshot,
  MinhDiagnosis,
  MinhMigrationPlan,
  MinhReport,
  RoiResult,
  TemperaturePrediction,
} from "./rackGuardianTypes";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_RACK_GUARDIAN_API_URL || "http://127.0.0.1:8001";
const ACTIONS: ActionType[] = [
  "increase_ventilation",
  "reduce_power_cap",
  "migrate_job",
  "no_action",
];
const REQUEST_TIMEOUT_MS = 4500;

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const timeoutController = new AbortController();
  const timeout = window.setTimeout(
    () => timeoutController.abort(),
    REQUEST_TIMEOUT_MS,
  );
  const abort = () => timeoutController.abort();
  signal?.addEventListener("abort", abort, { once: true });

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init.headers,
      },
      signal: timeoutController.signal,
    });
    if (!response.ok) throw new Error(`Backend returned ${response.status}`);
    return await response.json() as T;
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

function backendGpuId(gpu: GpuSnapshot): string {
  const rackNumber = Number.parseInt(gpu.rack_id.replace(/\D/g, ""), 10);
  const gpuNumber = Number.parseInt(gpu.gpu_id.replace(/\D/g, ""), 10);
  return `rack-${rackNumber}/gpu-${gpuNumber}`;
}

function normalizedStatus(status: string): CurrentGpu["status"] {
  if (status === "critical" || status === "warning") return status;
  return "safe";
}

function mockCurrent(gpu: GpuSnapshot): CurrentGpu {
  return {
    gpu_id: gpu.gpu_id,
    rack_id: gpu.rack_id,
    status: gpu.status,
    current_temp_c: gpu.temperature_c,
    predicted_temp_c: gpu.predicted_temperature_c,
    peak_temp_c: gpu.predicted_temperature_c,
    time_to_threshold_s: null,
    power_draw_w: gpu.power_draw_w,
    utilization_percent: gpu.utilization_percent,
  };
}

function mockPrediction(gpu: GpuSnapshot): TemperaturePrediction {
  return {
    gpu_id: gpu.gpu_id,
    rack_id: gpu.rack_id,
    source: "mock_fallback",
    current_temp_c: gpu.temperature_c,
    predicted_peak_temp_c: gpu.predicted_temperature_c,
    predicted_equilibrium_temp_c: gpu.predicted_temperature_c,
    safe_limit_c: 85,
    time_to_threshold_s: null,
    risk: gpu.status,
    confidence: null,
    trajectory: [],
  };
}

function mockSimulation(
  gpu: GpuSnapshot,
  actionType: ActionType,
): ActionSimulation {
  const isRecommended =
    actionType === "increase_ventilation" &&
    gpu.recommended_action.toLowerCase().includes("ventilation");
  const coolingGain = isRecommended
    ? gpu.simulated_solution.cooling_gain_c
    : actionType === "migrate_job"
      ? Math.max(gpu.simulated_solution.cooling_gain_c, 10)
      : actionType === "reduce_power_cap"
        ? 6
        : 0;
  const after = gpu.predicted_temperature_c - coolingGain;

  return {
    gpu_id: gpu.gpu_id,
    rack_id: gpu.rack_id,
    action_type: actionType,
    before: {
      risk: gpu.status,
      peak_temp_c: gpu.predicted_temperature_c,
      time_to_threshold_s: null,
    },
    after: {
      risk: after >= 85 ? "critical" : after >= 78 ? "warning" : "safe",
      peak_temp_c: after,
      convergence_temp_c: after,
      time_to_threshold_s: null,
    },
    cooling_gain_c: coolingGain,
  };
}

function mockRecommendation(gpu: GpuSnapshot): FinalRecommendation {
  const actionType: ActionType = gpu.recommended_action
    .toLowerCase()
    .includes("ventilation")
    ? "increase_ventilation"
    : gpu.status === "normal"
      ? "no_action"
      : "migrate_job";

  return {
    gpu_id: gpu.gpu_id,
    rack_id: gpu.rack_id,
    action_label: gpu.recommended_action,
    diagnosis: {
      likely_cause: gpu.risk_reason,
      confidence: null,
      reasons: [gpu.risk_reason],
    },
    why: [gpu.risk_reason],
    thermal_context: {
      current_temp_c: gpu.temperature_c,
      predicted_temp_no_action_c: gpu.predicted_temperature_c,
      predicted_temp_after_action_c:
        gpu.simulated_solution.temperature_after_c,
      cooling_gain_c: gpu.simulated_solution.cooling_gain_c,
      risk_no_action: gpu.risk_score,
      risk_after_action:
        gpu.simulated_solution.risk_after === "normal" ? 0.12 : 0.45,
    },
    simulation: mockSimulation(gpu, actionType),
    ranked_actions: mockActionScores(gpu),
    action_scores: mockActionScores(gpu),
    report: {
      incident_summary: gpu.risk_reason,
      likely_cause: gpu.risk_reason,
      evidence_used: [gpu.risk_reason],
      operator_next_step: gpu.recommended_action,
    },
    migration_plan: null,
    llm_report: {
      summary: `${gpu.recommended_action} This recommendation is using local demo data.`,
      provider: "mock_fallback",
    },
  };
}

function mockActionScores(gpu: GpuSnapshot): ActionScore[] {
  const coolingGain = gpu.simulated_solution.cooling_gain_c;
  return [
    {
      action_id: "increase_cooling_request",
      label: "Increase ventilation",
      score: gpu.status === "critical" ? 92 : 78,
      cooling_gain_c: coolingGain,
      risk_reduction: gpu.status === "critical" ? 0.64 : 0.3,
      operational_cost: "Low energy increase",
      performance_impact: "none",
    },
    {
      action_id: "apply_power_frequency_cap",
      label: "Reduce GPU power cap",
      score: gpu.status === "critical" ? 84 : 72,
      cooling_gain_c: Math.max(4, coolingGain - 2),
      risk_reduction: gpu.status === "critical" ? 0.52 : 0.24,
      operational_cost: "Moderate performance impact",
      performance_impact: "medium",
    },
    {
      action_id: "migrate_inference_traffic",
      label: "Migrate workload",
      score: gpu.status === "critical" ? 76 : 65,
      cooling_gain_c: Math.max(7, coolingGain),
      risk_reduction: gpu.status === "critical" ? 0.58 : 0.28,
      operational_cost: "Migration coordination required",
      performance_impact: "low",
    },
  ].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
}

export async function getBackendHealth(
  signal?: AbortSignal,
): Promise<BackendHealth> {
  try {
    return await requestJson<BackendHealth>("/health", {}, signal);
  } catch {
    return { status: "fallback", mock_fallback: true };
  }
}

export async function getCurrentGpus(
  signal?: AbortSignal,
): Promise<CurrentGpuResponse> {
  try {
    return await requestJson<CurrentGpuResponse>(
      "/api/gpus/current?rack_count=4&gpus_per_rack=8&seed=42",
      {},
      signal,
    );
  } catch {
    return {
      source: "mock_fallback",
      mock_fallback: true,
      racks: rackFleet.map((rack) => ({
        rack_id: rack.rack_id,
        status: rack.status,
        gpu_count: rack.gpu_count,
        gpus: rack.gpus.map(mockCurrent),
      })),
    };
  }
}

export async function getTemperaturePrediction(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<TemperaturePrediction> {
  try {
    return await requestJson<TemperaturePrediction>(
      "/api/predictions/temperature",
      {
        method: "POST",
        body: JSON.stringify({ gpu_id: backendGpuId(gpu), seed: 42 }),
      },
      signal,
    );
  } catch {
    return mockPrediction(gpu);
  }
}

export async function getActionSimulations(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<ActionSimulation[]> {
  return Promise.all(
    ACTIONS.map(async (actionType) => {
      try {
        return await requestJson<ActionSimulation>(
          "/api/simulations/actions",
          {
            method: "POST",
            body: JSON.stringify({
              gpu_id: backendGpuId(gpu),
              action_type: actionType,
              seed: 42,
            }),
          },
          signal,
        );
      } catch {
        return mockSimulation(gpu, actionType);
      }
    }),
  );
}

export async function getFinalRecommendation(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<FinalRecommendation> {
  try {
    return await requestJson<FinalRecommendation>(
      "/api/recommendations/final",
      {
        method: "POST",
        body: JSON.stringify({ gpu_id: backendGpuId(gpu), seed: 42 }),
      },
      signal,
    );
  } catch {
    return mockRecommendation(gpu);
  }
}

export async function getMinhDiagnosis(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<MinhDiagnosis> {
  const result = await getFinalRecommendation(gpu, signal);
  return result.diagnosis ?? mockRecommendation(gpu).diagnosis!;
}

export async function getMinhRecommendation(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<FinalRecommendation> {
  return getFinalRecommendation(gpu, signal);
}

export async function getMinhForecast(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<TemperaturePrediction> {
  return getTemperaturePrediction(gpu, signal);
}

export async function getMinhActionScores(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<ActionScore[]> {
  const result = await getFinalRecommendation(gpu, signal);
  return result.action_scores ?? result.ranked_actions ?? [];
}

export async function getMinhReport(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<MinhReport> {
  const result = await getFinalRecommendation(gpu, signal);
  return result.report ?? mockRecommendation(gpu).report!;
}

export async function getMinhMigrationPlan(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<MinhMigrationPlan> {
  const result = await getFinalRecommendation(gpu, signal);
  return result.migration_plan ?? buildFallbackMigrationPlan(gpu, result);
}

function buildFallbackMigrationPlan(
  gpu: GpuSnapshot,
  result: FinalRecommendation,
): MinhMigrationPlan {
  return {
    simulated: true,
    target_action: result.action_label || gpu.recommended_action,
    reason:
      result.diagnosis?.likely_cause ??
      "Thermal risk requires an operator-reviewed mitigation plan.",
    steps: [
      "Confirm destination GPU capacity.",
      "Pause or checkpoint the workload.",
      "Simulate the workload transfer.",
      "Verify the predicted thermal state.",
    ],
    estimated_risk_reduction: result.roi_result?.risk_reduction ?? null,
    operator_confirmation_required: true,
  };
}

export async function getRoi(
  gpu: GpuSnapshot,
  recommendation: FinalRecommendation,
  signal?: AbortSignal,
): Promise<RoiResult | null> {
  const existing = recommendation.roi_result;
  const actionType = existing?.action_type;
  if (!actionType) return null;

  const inputs = recommendation.assumptions?.financial_inputs?.[actionType];
  const thermal = recommendation.thermal_context;
  if (!inputs || thermal?.risk_no_action == null || thermal.risk_after_action == null) {
    return existing ?? null;
  }

  try {
    return await requestJson<RoiResult>(
      "/api/roi",
      {
        method: "POST",
        body: JSON.stringify({
          action_type: actionType,
          risk_no_action: thermal.risk_no_action,
          risk_after_action: thermal.risk_after_action,
          ...inputs,
        }),
      },
      signal,
    );
  } catch {
    return existing ?? mockRecommendation(gpu).roi_result ?? null;
  }
}

export async function loadGpuDecision(
  gpu: GpuSnapshot,
  signal?: AbortSignal,
): Promise<GpuDecisionData> {
  const [health, currentResponse, prediction, simulations, recommendation] =
    await Promise.all([
      getBackendHealth(signal),
      getCurrentGpus(signal),
      getTemperaturePrediction(gpu, signal),
      getActionSimulations(gpu, signal),
      getFinalRecommendation(gpu, signal),
    ]);
  const current =
    currentResponse.racks
      .flatMap((rack) => rack.gpus)
      .find(
        (item) =>
          Number.parseInt(item.rack_id.replace(/\D/g, ""), 10) ===
            Number.parseInt(gpu.rack_id.replace(/\D/g, ""), 10) &&
          Number.parseInt(item.gpu_id.replace(/\D/g, ""), 10) ===
            Number.parseInt(gpu.gpu_id.replace(/\D/g, ""), 10),
      ) ?? mockCurrent(gpu);
  current.status = normalizedStatus(current.status);
  const roi = await getRoi(gpu, recommendation, signal);
  recommendation.migration_plan ??= buildFallbackMigrationPlan(
    gpu,
    recommendation,
  );
  const hasBackendData =
    health.status === "ok" &&
    currentResponse.source !== "mock_fallback" &&
    prediction.source !== "mock_fallback" &&
    recommendation.llm_report?.provider !== "mock_fallback";

  return {
    current,
    prediction,
    simulations,
    recommendation,
    roi,
    source: hasBackendData ? "backend" : "fallback",
  };
}
