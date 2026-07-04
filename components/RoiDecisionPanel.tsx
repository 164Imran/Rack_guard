"use client";

import { useEffect, useState } from "react";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  CircleDollarSign,
  Gauge,
  RefreshCw,
  ShieldAlert,
  Thermometer,
} from "lucide-react";

import styles from "./RoiDecisionPanel.module.css";

type DecisionStatus = "recommended" | "partial" | "not_recommended";

type RoiDecisionData = {
  gpu_id?: number;
  action_label?: string;
  thermal_context?: {
    current_temp_c?: number;
    predicted_temp_no_action_c?: number;
    predicted_temp_after_action_c?: number;
    cooling_gain_c?: number;
    risk_no_action?: number;
    risk_after_action?: number;
    fan_speed_percent?: number;
    power_draw_w?: number;
    gpu_utilization_percent?: number;
  };
  roi_result?: {
    action_type?: string;
    risk_reduction?: number;
    avoided_loss_eur?: number;
    action_cost_eur?: number;
    net_gain_eur?: number;
    roi_percent?: number | null;
    decision?: DecisionStatus;
    gpu_hour_value_eur?: number;
    job_remaining_hours?: number;
  };
  hidden_cost_audit?: {
    missing_assumptions?: string[];
    hidden_costs?: string[];
    operational_risks?: string[];
    feasibility_warnings?: string[];
    confidence_level?: string;
    should_recalculate_roi?: boolean;
    suggested_roi_inputs_to_add?: string[];
    operator_summary?: string;
    manager_summary?: string;
  };
  llm_report?: {
    summary?: string;
  };
};

const mockDecision: RoiDecisionData = {
  gpu_id: 0,
  action_label: "Increase ventilation +20%",
  thermal_context: {
    current_temp_c: 74,
    predicted_temp_no_action_c: 84,
    predicted_temp_after_action_c: 76,
    cooling_gain_c: 8,
    risk_no_action: 0.82,
    risk_after_action: 0.18,
    fan_speed_percent: 92,
    power_draw_w: 310,
    gpu_utilization_percent: 96,
  },
  roi_result: {
    action_type: "increase_ventilation",
    risk_reduction: 0.64,
    avoided_loss_eur: 1.152,
    action_cost_eur: 0.15,
    net_gain_eur: 1.002,
    roi_percent: 668,
    decision: "recommended",
    gpu_hour_value_eur: 10,
    job_remaining_hours: 4,
  },
  hidden_cost_audit: {
    missing_assumptions: ["ambient_temp_c", "cooling_headroom_percent"],
    hidden_costs: ["fan wear", "additional HVAC energy consumption"],
    operational_risks: [
      "Fan speed is already high, so real cooling gain may be lower than simulated.",
    ],
    feasibility_warnings: [
      "Ventilation increase may be limited by available fan capacity.",
    ],
    confidence_level: "medium",
    should_recalculate_roi: true,
    suggested_roi_inputs_to_add: [
      "fan_wear_cost_eur",
      "extra_hvac_power_kw",
    ],
    operator_summary:
      "The action is promising, but cooling headroom and ambient temperature are missing. Validate fan capacity before applying.",
    manager_summary:
      "The ROI looks positive, but confidence is medium because hidden cooling costs and physical cooling limits are not fully modeled.",
  },
  llm_report: {
    summary:
      "Increasing ventilation is currently the most profitable mitigation action, but the ROI should be validated with cooling headroom and ambient temperature data.",
  },
};

const decisionApiUrl = process.env.NEXT_PUBLIC_DECISION_API_URL;

export default function RoiDecisionPanel() {
  const [data, setData] = useState<RoiDecisionData>(mockDecision);
  const [status, setStatus] = useState<"loading" | "ready" | "fallback">(
    decisionApiUrl ? "loading" : "fallback",
  );

  useEffect(() => {
    if (!decisionApiUrl) return;
    const controller = new AbortController();

    async function loadDecision() {
      try {
        const response = await fetch(decisionApiUrl as string, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Decision API unavailable");
        setData(await response.json());
        setStatus("ready");
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          setData(mockDecision);
          setStatus("fallback");
        }
      }
    }

    void loadDecision();
    return () => controller.abort();
  }, []);

  if (status === "loading") {
    return (
      <div className={styles.loading} role="status">
        <RefreshCw aria-hidden="true" />
        <span>Loading ROI decision pipeline...</span>
      </div>
    );
  }

  const thermal = data.thermal_context;
  const roi = data.roi_result;
  const audit = data.hidden_cost_audit;
  const decision = roi?.decision ?? "not_recommended";
  const actionName =
    data.action_label ?? humanize(roi?.action_type) ?? "Not provided";

  return (
    <section className={styles.panel} aria-label="ROI decision pipeline">
      <header className={styles.verdict}>
        <div className={styles.verdictIcon}>
          <CircleDollarSign aria-hidden="true" />
        </div>
        <div className={styles.verdictCopy}>
          <span>Recommended action</span>
          <h2>{actionName}</h2>
          <p>
            GPU {formatValue(data.gpu_id)} · Thermal mitigation with financial
            validation
          </p>
        </div>
        <div className={styles.statusGroup}>
          <span className={`${styles.decision} ${styles[decision]}`}>
            {humanize(decision)}
          </span>
          <span className={styles.confidence}>
            {audit?.confidence_level ?? "Not provided"} confidence
          </span>
          {status === "fallback" && (
            <span className={styles.demoBadge}>Demo data</span>
          )}
        </div>
      </header>

      <div className={styles.columns}>
        <div className={styles.column}>
          <article className={styles.card}>
            <CardTitle icon={Thermometer} title="Thermal impact" />
            <div className={styles.temperatureFlow}>
              <Metric
                label="Without action"
                value={formatTemperature(thermal?.predicted_temp_no_action_c)}
                tone="danger"
              />
              <span aria-hidden="true">→</span>
              <Metric
                label="After action"
                value={formatTemperature(thermal?.predicted_temp_after_action_c)}
                tone="safe"
              />
              <Metric
                label="Cooling gain"
                value={
                  thermal?.cooling_gain_c == null
                    ? "Not provided"
                    : `-${formatNumber(thermal.cooling_gain_c)}°C`
                }
                tone="accent"
              />
            </div>
            <div className={styles.compactMetrics}>
              <InlineMetric
                label="Current"
                value={formatTemperature(thermal?.current_temp_c)}
              />
              <InlineMetric
                label="Risk before"
                value={formatProbability(thermal?.risk_no_action)}
              />
              <InlineMetric
                label="Risk after"
                value={formatProbability(thermal?.risk_after_action)}
              />
            </div>
          </article>

          <article className={`${styles.card} ${styles.financialCard}`}>
            <CardTitle icon={Gauge} title="Financial impact" />
            <div className={styles.financialHero}>
              <Metric
                label="Net gain"
                value={formatCurrency(roi?.net_gain_eur, true)}
                tone="safe"
              />
              <Metric
                label="ROI"
                value={formatRoi(roi?.roi_percent)}
                tone="safe"
              />
            </div>
            <div className={styles.compactMetrics}>
              <InlineMetric
                label="Action cost"
                value={formatCurrency(roi?.action_cost_eur)}
              />
              <InlineMetric
                label="Avoided loss"
                value={formatCurrency(roi?.avoided_loss_eur)}
              />
              <InlineMetric
                label="GPU hour"
                value={formatCurrency(roi?.gpu_hour_value_eur)}
              />
              <InlineMetric
                label="Job remaining"
                value={formatHours(roi?.job_remaining_hours)}
              />
            </div>
          </article>
        </div>

        <div className={styles.column}>
          <article className={`${styles.card} ${styles.auditCard}`}>
            <div className={styles.auditHeading}>
              <CardTitle icon={ShieldAlert} title="Hidden cost audit" />
              <span
                className={
                  audit?.should_recalculate_roi
                    ? styles.recalculate
                    : styles.validated
                }
              >
                {audit?.should_recalculate_roi
                  ? "ROI update needed"
                  : "Inputs validated"}
              </span>
            </div>
            <AuditRow
              label="Missing assumptions"
              values={audit?.missing_assumptions}
              warning
            />
            <AuditRow label="Hidden costs" values={audit?.hidden_costs} />
            <AuditRow
              label="Operational risks"
              values={audit?.operational_risks}
            />
            <AuditRow
              label="Feasibility"
              values={audit?.feasibility_warnings}
              warning
            />
          </article>

          <article className={`${styles.card} ${styles.reportCard}`}>
            <CardTitle icon={Bot} title="Final Crusoe report" />
            <p>{data.llm_report?.summary ?? "Not provided"}</p>
            <div className={styles.reportSummaries}>
              <div>
                <span>Operator</span>
                <p>{audit?.operator_summary ?? "Not provided"}</p>
              </div>
              <div>
                <span>Manager</span>
                <p>{audit?.manager_summary ?? "Not provided"}</p>
              </div>
            </div>
          </article>
        </div>
      </div>
    </section>
  );
}

function CardTitle({
  icon: Icon,
  title,
}: {
  icon: typeof CheckCircle2;
  title: string;
}) {
  return (
    <div className={styles.cardTitle}>
      <Icon aria-hidden="true" />
      <h3>{title}</h3>
    </div>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "safe" | "danger" | "accent";
}) {
  return (
    <div className={`${styles.metric} ${tone ? styles[tone] : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InlineMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.inlineMetric}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AuditRow({
  label,
  values,
  warning = false,
}: {
  label: string;
  values?: string[];
  warning?: boolean;
}) {
  const items = values?.length ? values : ["No issue detected"];
  return (
    <div className={styles.auditRow}>
      <span>
        {warning ? <AlertTriangle aria-hidden="true" /> : null}
        {label}
      </span>
      <p>{items.join(" · ")}</p>
    </div>
  );
}

function formatValue(value: unknown) {
  return value == null ? "Not provided" : String(value);
}

function formatNumber(value?: number) {
  return value == null
    ? "Not provided"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

function formatCurrency(value?: number, showSign = false) {
  if (value == null) return "Not provided";
  const formatted = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "EUR",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(Math.abs(value));
  if (!showSign || value === 0) return value < 0 ? `-${formatted}` : formatted;
  return `${value > 0 ? "+" : "-"}${formatted}`;
}

function formatRoi(value?: number | null) {
  if (value == null) return "N/A";
  return `${value > 0 ? "+" : ""}${formatNumber(value)}%`;
}

function formatProbability(value?: number) {
  return value == null ? "Not provided" : `${formatNumber(value * 100)}%`;
}

function formatTemperature(value?: number) {
  return value == null ? "Not provided" : `${formatNumber(value)}°C`;
}

function formatHours(value?: number) {
  return value == null ? "Not provided" : `${formatNumber(value)} h`;
}

function humanize(value?: string) {
  if (!value) return undefined;
  return value.replaceAll("_", " ").replace(/^\w/, (letter) =>
    letter.toUpperCase(),
  );
}
