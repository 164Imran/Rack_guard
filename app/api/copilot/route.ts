import { NextResponse } from "next/server";

const DEFAULT_BASE_URL = "https://api.inference.crusoecloud.com/v1";
const DEFAULT_MODEL = "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B";
const SYSTEM_PROMPT =
  "You are Thermal ROI Copilot, an AI assistant for small GPU clusters. Answer concisely. Use the provided telemetry context. Do not invent unavailable numbers. Focus on thermal risk, GPU status, recommended checks, and operational clarity.";

type CopilotContext = {
  selectedRack?: Record<string, unknown> | null;
  selectedGpu?: Record<string, unknown> | null;
  fleetSummary?: Record<string, unknown> | null;
};

type CopilotRequest = {
  message?: unknown;
  context?: CopilotContext;
};

function value(record: Record<string, unknown>, key: string, fallback = "not provided") {
  const result = record[key];
  return result === undefined || result === null ? fallback : String(result);
}

function fallbackCopilotAnswer(message: string, context: CopilotContext): string {
  const question = message.toLowerCase();
  const gpu = context.selectedGpu;
  const rack = context.selectedRack;
  const fleet = context.fleetSummary;

  if (gpu) {
    const gpuId = value(gpu, "gpu_id", "Selected GPU");
    const status = value(gpu, "status");
    const current = value(gpu, "temperature_c");
    const predicted = value(gpu, "predicted_temperature_c");
    const reason = value(gpu, "risk_reason", "The available telemetry indicates elevated thermal risk.");
    const action = value(gpu, "recommended_action", "Inspect cooling and workload pressure.");

    if (question.includes("pourquoi") || question.includes("why") || question.includes("critique")) {
      return `${gpuId} is ${status}. ${reason} Current temperature is ${current}°C and predicted temperature is ${predicted}°C.`;
    }
    if (question.includes("action") || question.includes("faire") || question.includes("dois")) {
      return `Priority action for ${gpuId}: ${action}. Verify cooling flow, workload pressure, and power draw before applying changes.`;
    }
    return `${gpuId} is ${status}. Current temperature is ${current}°C and predicted temperature is ${predicted}°C. Priority: ${action}.`;
  }

  if (rack) {
    const rackId = value(rack, "rack_id", "Selected rack");
    const toCheck = value(rack, "gpus_to_check", "0");
    const critical = value(rack, "critical_count", "0");
    const warning = value(rack, "warning_count", "0");
    return `${rackId} should be checked. ${toCheck} GPUs require review: ${critical} critical and ${warning} warning. Inspect critical GPUs first.`;
  }

  if (fleet) {
    const highestRiskGpu = value(fleet, "highest_risk_gpu", "not identified");
    const highestRiskRack = value(fleet, "highest_risk_rack", "not identified");
    const racksToCheck = value(fleet, "racks_to_check", "0");
    const gpusToCheck = value(fleet, "gpus_to_check", "0");

    if (question.includes("gpu") || question.includes("risque")) {
      return `${highestRiskGpu} is currently the highest-risk GPU in ${highestRiskRack}. Review its telemetry and cooling conditions first.`;
    }
    if (question.includes("rack") || question.includes("vérifier")) {
      return `${highestRiskRack} is the priority rack. The fleet currently has ${racksToCheck} racks and ${gpusToCheck} GPUs requiring review.`;
    }
    return `Cluster summary: ${racksToCheck} racks and ${gpusToCheck} GPUs require review. Highest priority is ${highestRiskRack}, starting with ${highestRiskGpu}.`;
  }

  return "No telemetry context is available yet. Select a rack or GPU, then ask me to assess its thermal risk.";
}

export async function POST(request: Request) {
  let payload: CopilotRequest;

  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON payload." }, { status: 400 });
  }

  const message = typeof payload.message === "string" ? payload.message.trim() : "";
  const context = payload.context ?? {};

  if (!message) {
    return NextResponse.json({ ok: false, error: "Message is required." }, { status: 400 });
  }

  const fallback = fallbackCopilotAnswer(message, context);
  const apiKey = process.env.CRUSOE_API_KEY;
  const baseUrl = (process.env.CRUSOE_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, "");
  const model = process.env.CRUSOE_MODEL || DEFAULT_MODEL;

  if (!apiKey) {
    return NextResponse.json({
      ok: true,
      answer: fallback,
      source: "fallback",
      model,
      error: "CRUSOE_API_KEY is not configured.",
    });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12_000);

  try {
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          {
            role: "user",
            content: `Context JSON: ${JSON.stringify(context)}\n\nUser question: ${message}`,
          },
        ],
        temperature: 0.2,
        max_tokens: 350,
      }),
      signal: controller.signal,
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`Crusoe request failed with status ${response.status}.`);
    }

    const data = await response.json() as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const answer = data.choices?.[0]?.message?.content?.trim();

    if (!answer) {
      throw new Error("Crusoe returned an empty answer.");
    }

    return NextResponse.json({ ok: true, answer, source: "crusoe", model });
  } catch (error) {
    const messageText = error instanceof Error && error.name === "AbortError"
      ? "Crusoe request timed out."
      : error instanceof Error
        ? error.message
        : "Crusoe request failed.";

    return NextResponse.json({
      ok: true,
      answer: fallback,
      source: "fallback",
      model,
      error: messageText,
    });
  } finally {
    clearTimeout(timeout);
  }
}
