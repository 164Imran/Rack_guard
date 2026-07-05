export type GpuStatus = "normal" | "safe" | "warning" | "critical";
export type ActionType =
  | "increase_ventilation"
  | "reduce_power_cap"
  | "migrate_job"
  | "no_action";

export type GpuSnapshot = {
  gpu_id: string;
  rack_id: string;
  status: Exclude<GpuStatus, "safe">;
  temperature_c: number;
  predicted_temperature_c: number;
  utilization_percent: number;
  power_draw_w: number;
  risk_score: number;
  risk_reason: string;
  recommended_action: string;
  simulated_solution: {
    action: string;
    temperature_after_c: number;
    cooling_gain_c: number;
    risk_after: Exclude<GpuStatus, "safe">;
    performance_impact: "none" | "low" | "medium";
  };
};

export type RackSnapshot = {
  rack_id: string;
  gpu_count: number;
  gpus_to_check: number;
  normal_count: number;
  warning_count: number;
  critical_count: number;
  status: Exclude<GpuStatus, "safe">;
  gpus: GpuSnapshot[];
  cause: string;
  secondary_cause: string;
  recommendation: string;
  slowdown: string;
};

export type BackendHealth = {
  status: string;
  service?: string;
  modules?: Record<string, string>;
  mock_fallback?: boolean;
};

export type CurrentGpu = {
  gpu_id: string;
  rack_id: string;
  status: GpuStatus;
  current_temp_c: number | null;
  predicted_temp_c: number | null;
  peak_temp_c?: number | null;
  time_to_threshold_s?: number | null;
  power_draw_w: number | null;
  utilization_percent: number | null;
};

export type CurrentGpuResponse = {
  source: string;
  mock_fallback: boolean;
  racks: Array<{
    rack_id: string;
    status: GpuStatus;
    current_temp_c?: number | null;
    gpu_count: number;
    gpus: CurrentGpu[];
  }>;
};

export type TemperaturePrediction = {
  gpu_id: string;
  rack_id: string;
  source: string;
  current_temp_c: number | null;
  predicted_peak_temp_c: number | null;
  predicted_equilibrium_temp_c: number | null;
  safe_limit_c: number | null;
  time_to_threshold_s: number | null;
  risk: GpuStatus;
  confidence: number | null;
  trajectory: Array<number | { t_s?: number; gpu_temp_c?: number }>;
};

export type ActionSimulation = {
  gpu_id: string;
  rack_id: string;
  action_type: ActionType;
  source_action_id?: string;
  before: {
    risk?: GpuStatus;
    peak_temp_c?: number | null;
    time_to_threshold_s?: number | null;
  };
  after: {
    risk?: GpuStatus;
    peak_temp_c?: number | null;
    convergence_temp_c?: number | null;
    time_to_threshold_s?: number | null;
  };
  cooling_gain_c?: number | null;
};

export type RoiResult = {
  action_type: ActionType;
  risk_reduction: number;
  avoided_loss_eur: number;
  action_cost_eur: number;
  net_gain_eur: number;
  roi_percent: number | null;
  decision: "recommended" | "partial" | "not_recommended";
  gpu_hour_value_eur?: number;
  job_remaining_hours?: number;
};

export type ActionScore = {
  action_id?: string;
  action_type?: ActionType;
  label?: string;
  score?: number;
  expected_impact?: string;
  rationale?: string;
  operational_cost?: string;
  net_gain_eur?: number;
  roi_percent?: number | null;
  decision?: string;
};

export type FinalRecommendation = {
  gpu_id: string;
  rack_id: string;
  action_label: string;
  diagnosis?: {
    likely_cause?: string;
    confidence?: number | null;
    reasons?: string[];
    evidence_summary?: string;
  };
  why?: string[];
  thermal_context?: {
    current_temp_c?: number | null;
    predicted_temp_no_action_c?: number | null;
    predicted_temp_after_action_c?: number | null;
    cooling_gain_c?: number | null;
    risk_no_action?: number;
    risk_after_action?: number;
  };
  simulation?: ActionSimulation;
  roi_result?: RoiResult;
  ranked_actions?: ActionScore[];
  action_scores?: ActionScore[];
  report?: {
    incident_summary?: string;
    likely_cause?: string;
    evidence_used?: string[];
    operator_next_step?: string;
    escalation_note?: string;
  };
  migration_plan?: Record<string, unknown> | null;
  llm_report?: { summary?: string; provider?: string };
  assumptions?: {
    financial_inputs?: Partial<Record<ActionType, Record<string, number>>>;
  };
};

export type GpuDecisionData = {
  current: CurrentGpu;
  prediction: TemperaturePrediction;
  simulations: ActionSimulation[];
  recommendation: FinalRecommendation;
  roi: RoiResult | null;
  source: "backend" | "fallback";
};
