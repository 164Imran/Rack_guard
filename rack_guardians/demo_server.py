"""
Local Rack Guardian demo server.

Run from the Rack_guard folder:
    C:\\Users\\phamq\\anaconda3\\python.exe -B rack_guardians\\demo_server.py --port 3000

The server uses only the Python standard library for HTTP. The thermal data
comes from Bloc A's simulator through agent.build_gpu_fleet().
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agent import (  # noqa: E402
    DEFAULT_THRESHOLD_C,
    ForecastSummary,
    RackTelemetry,
    build_demo_case,
    build_gpu_fleet,
    build_inference_fleet,
    build_simulated_fleet,
    classify_risk,
    evaluate_rack,
    execute_migration_plan,
    prediction_table,
    propose_migration_plan,
    simulate_mitigation,
    telemetry_table,
)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rack Guardian Control Room</title>
  <style>
    :root {
      --bg: #f4f6f2;
      --surface: #ffffff;
      --surface-2: #f9faf7;
      --ink: #18201d;
      --muted: #657069;
      --line: #d7ddd3;
      --teal: #087f8c;
      --blue: #246bfe;
      --green: #2e7d32;
      --amber: #b7791f;
      --red: #b42318;
      --violet: #6750a4;
      --shadow: 0 14px 34px rgba(24, 32, 29, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, textarea, input { font: inherit; }
    button {
      min-height: 38px;
      border: 0;
      border-radius: 6px;
      background: var(--teal);
      color: #fff;
      padding: 9px 12px;
      font-weight: 750;
      cursor: pointer;
    }
    button.secondary {
      background: #eef4f2;
      color: #1d5258;
      border: 1px solid #c9dbd9;
    }
    button.icon {
      width: 38px;
      min-width: 38px;
      padding: 0;
      font-size: 24px;
      line-height: 1;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 0 22px;
      background: rgba(255, 255, 255, 0.95);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 800;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 780;
    }
    h3 {
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 780;
    }
    .subtle { color: var(--muted); }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .mode-switch {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-2);
    }
    .mode-btn {
      min-height: 32px;
      padding: 7px 10px;
      background: transparent;
      color: var(--muted);
      border: 0;
    }
    .mode-btn.active {
      background: var(--teal);
      color: #fff;
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 7px;
      color: var(--muted);
      font-weight: 650;
      white-space: nowrap;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(340px, 390px);
      gap: 18px;
      padding: 18px;
      max-width: 1560px;
      margin: 0 auto;
    }
    .main {
      display: grid;
      gap: 18px;
      min-width: 0;
    }
    .panel, .chat {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel { padding: 16px; }
    .real-input-panel {
      display: grid;
      gap: 14px;
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }
    .field {
      display: grid;
      gap: 5px;
    }
    .field label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      text-transform: uppercase;
    }
    .field input, .field textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      outline: none;
    }
    .dataset-box {
      min-height: 122px;
      resize: vertical;
    }
    .panel-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 12px;
    }
    .metric {
      background: var(--surface-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .metric .value {
      margin-top: 8px;
      font-size: 22px;
      font-weight: 820;
      overflow-wrap: anywhere;
    }
    .rack-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 12px;
    }
    .rack {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 150px;
      background: #fff;
      cursor: pointer;
      text-align: left;
      color: var(--ink);
      box-shadow: none;
      position: relative;
    }
    .rack-card {
      min-height: 210px;
      padding-left: 22px;
      display: grid;
      grid-template-rows: 1fr 1fr;
      gap: 10px;
    }
    .gpu-card {
      min-height: 150px;
      border-left-width: 6px;
    }
    .dual-strip {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 8px;
      display: grid;
      grid-template-rows: 1fr 1fr;
      overflow: hidden;
      border-radius: 8px 0 0 8px;
    }
    .strip-segment.SAFE { background: var(--green); }
    .strip-segment.MEDIUM, .strip-segment.WATCH { background: var(--amber); }
    .strip-segment.CRITICAL, .strip-segment.HIGH { background: var(--red); }
    .rack.selected {
      outline: 3px solid rgba(8, 127, 140, 0.22);
      border-color: #8ccfd5;
    }
    .rack.SAFE { border-left-color: var(--green); }
    .rack.MEDIUM, .rack.WATCH { border-left-color: var(--amber); }
    .rack.HIGH, .rack.CRITICAL { border-left-color: var(--red); }
    .rack-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .rack-upper {
      display: grid;
      align-content: center;
      gap: 6px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }
    .rack-lower {
      display: grid;
      align-content: start;
      gap: 8px;
    }
    .rack-id { font-size: 16px; font-weight: 820; }
    .risk {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 820;
    }
    .risk.SAFE { background: #e8f5e9; color: var(--green); }
    .risk.MEDIUM, .risk.WATCH { background: #fff8e1; color: var(--amber); }
    .risk.HIGH, .risk.CRITICAL { background: #fde8e7; color: var(--red); }
    .temp {
      margin-top: 12px;
      font-size: 30px;
      font-weight: 840;
    }
    .rack-lines {
      margin-top: 8px;
      color: var(--muted);
      display: grid;
      gap: 3px;
    }
    .nav-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .gpu-mini-grid {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 5px;
      margin-top: 12px;
    }
    .gpu-mini {
      height: 18px;
      border-radius: 4px;
      border: 1px solid var(--line);
      background: #e8f5e9;
    }
    .gpu-mini.WATCH { background: #fff3cd; }
    .gpu-mini.HIGH, .gpu-mini.CRITICAL { background: #fde8e7; }
    .tables {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      background: var(--surface-2);
      font-size: 12px;
      text-transform: uppercase;
    }
    tr:last-child td { border-bottom: 0; }
    .chat {
      min-height: calc(100vh - 100px);
      max-height: calc(100vh - 100px);
      display: grid;
      grid-template-rows: auto 1fr auto;
      position: sticky;
      top: 82px;
      overflow: hidden;
    }
    .chat-header {
      padding: 15px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .messages {
      padding: 14px;
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 12px;
      background: #fbfcfa;
    }
    .message {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      background: #fff;
    }
    .message.user {
      border-color: #c9dbd9;
      background: #f0f8f7;
    }
    .message-title {
      font-weight: 780;
      margin-bottom: 5px;
    }
    .chat-compose {
      padding: 12px;
      border-top: 1px solid var(--line);
      background: #fff;
    }
    textarea {
      width: 100%;
      min-height: 74px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      color: var(--ink);
    }
    .compose-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
    }
    .attachments {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
      min-height: 18px;
      margin-top: 7px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 6px 9px;
      color: #2f3a35;
      font-size: 12px;
      font-weight: 700;
    }
    .link-btn {
      background: transparent;
      border: 0;
      color: var(--teal);
      padding: 0;
      min-height: 0;
      font-weight: 800;
      cursor: pointer;
    }
    .link-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 12px 0 0;
    }
    .message.plan {
      border-color: #bdd5ff;
      background: #f7faff;
    }
    .message.success {
      border-color: #b8dcbc;
      background: #f3fbf4;
    }
    .message.warning {
      border-color: #f1d19a;
      background: #fffaf0;
    }
    button.approve-btn {
      margin-top: 10px;
      background: var(--blue);
    }
    .guardrails {
      margin: 10px 0 0;
      padding-left: 18px;
    }
    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      z-index: 20;
      background: rgba(20, 28, 25, 0.42);
      padding: 26px;
    }
    .modal-backdrop.open { display: grid; place-items: center; }
    .modal {
      width: min(1080px, 100%);
      max-height: min(820px, calc(100vh - 52px));
      overflow: hidden;
      display: grid;
      grid-template-rows: auto auto 1fr;
      border-radius: 8px;
      background: var(--surface);
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
      border: 1px solid var(--line);
    }
    .modal-head {
      padding: 15px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .tabs {
      display: flex;
      gap: 8px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--surface-2);
      flex-wrap: wrap;
    }
    .tab {
      background: #fff;
      color: #2d3833;
      border: 1px solid var(--line);
      min-height: 34px;
    }
    .tab.active {
      background: var(--teal);
      color: #fff;
      border-color: var(--teal);
    }
    .modal-body {
      padding: 16px;
      overflow: auto;
    }
    canvas {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 7;
      min-height: 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .info-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .action-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      cursor: pointer;
    }
    .action-card.open {
      border-color: #9ccfd4;
      background: #f5fbfb;
    }
    .hidden-detail { display: none; }
    .action-card.open .hidden-detail { display: block; }
    .action-score {
      color: var(--violet);
      font-weight: 840;
    }
    .report-list {
      display: grid;
      gap: 12px;
    }
    .report-block {
      border-left: 4px solid var(--teal);
      border-radius: 6px;
      padding: 10px 12px;
      background: #f2faf9;
    }
    .report-block.alert {
      border-left-color: var(--red);
      background: #fde8e7;
      color: #681f1a;
    }
    .hidden { display: none; }
    @media (max-width: 1080px) {
      .app { grid-template-columns: 1fr; }
      .chat { position: static; min-height: 560px; max-height: none; }
    }
    @media (max-width: 760px) {
      header { height: auto; align-items: flex-start; flex-direction: column; padding: 14px 16px; }
      .summary, .tables, .form-grid { grid-template-columns: 1fr; }
      .app { padding: 12px; }
      .rack-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Rack Guardian</h1>
      <div class="subtle">Multi-GPU rack overheating forecast, diagnosis, and mitigation demo</div>
    </div>
    <div class="top-actions">
      <div class="mode-switch" aria-label="Demo mode">
        <button id="simulationModeBtn" class="mode-btn active" type="button">Simulation</button>
        <button id="realModeBtn" class="mode-btn" type="button">Real Input</button>
      </div>
      <label class="toggle"><input id="useCrusoe" type="checkbox" checked /> Nemotron evidence review</label>
      <button id="simulateBtn">Simulate Inference Workload</button>
      <span id="status" class="subtle">Ready</span>
    </div>
  </header>

  <div class="app">
    <div class="main">
      <section class="panel">
        <h2>Fleet Snapshot</h2>
        <div id="summary" class="summary"></div>
      </section>

      <section id="realInputPanel" class="panel real-input-panel hidden">
        <div class="nav-row">
          <div>
            <h2>Real Input</h2>
            <div class="subtle">Manual telemetry or JSON/CSV dataset</div>
          </div>
          <div class="panel-actions">
            <button id="loadRealDemoBtn" class="secondary" type="button">Load Example</button>
            <button id="analyzeRealBtn" type="button">Analyze Real Input</button>
          </div>
        </div>
        <div class="form-grid">
          <div class="field"><label for="realRackId">Rack</label><input id="realRackId" value="rack-1" /></div>
          <div class="field"><label for="realGpuId">GPU</label><input id="realGpuId" value="gpu-1" /></div>
          <div class="field"><label for="realGpuTemp">GPU Temp C</label><input id="realGpuTemp" type="number" step="0.1" value="62" /></div>
          <div class="field"><label for="realAssignedTraffic">Assigned Traffic %</label><input id="realAssignedTraffic" type="number" step="1" value="84" /></div>
          <div class="field"><label for="realQueue">Queue Length</label><input id="realQueue" type="number" step="1" value="96" /></div>
          <div class="field"><label for="realPower">GPU Power W</label><input id="realPower" type="number" step="1" value="335" /></div>
          <div class="field"><label for="realSmUtil">SM Util %</label><input id="realSmUtil" type="number" step="1" value="91" /></div>
          <div class="field"><label for="realMemUtil">Memory Util %</label><input id="realMemUtil" type="number" step="1" value="68" /></div>
          <div class="field"><label for="realCoolingFlow">Cooling Flow LPM</label><input id="realCoolingFlow" type="number" step="0.01" value="0.58" /></div>
          <div class="field"><label for="realFanSpeed">Fan Speed RPM</label><input id="realFanSpeed" type="number" step="1" value="5100" /></div>
          <div class="field"><label for="realLatency">Network Latency MS</label><input id="realLatency" type="number" step="0.1" value="18" /></div>
          <div class="field"><label for="realHeadroom">Safe Headroom %</label><input id="realHeadroom" type="number" step="1" value="18" /></div>
        </div>
        <div class="field">
          <label for="realDataset">Dataset</label>
          <textarea id="realDataset" class="dataset-box" placeholder='[{"rack_id":"rack-1","gpu_id":"gpu-1","gpu_temp_c":62,"assigned_traffic_pct":84,"inference_queue_len":96}]'></textarea>
        </div>
        <div class="panel-actions">
          <input id="realDatasetFile" type="file" accept=".json,.csv,text/csv,application/json" />
          <button id="clearRealBtn" class="secondary" type="button">Clear Real Input</button>
        </div>
      </section>

      <section class="panel">
        <div class="nav-row">
          <div>
            <h2 id="mapTitle">Rack Map</h2>
            <div id="mapSub" class="subtle">3 racks, 8 GPUs per rack</div>
          </div>
          <button id="backBtn" class="secondary hidden">Back To Racks</button>
        </div>
        <div id="rackGrid" class="rack-grid"></div>
      </section>

      <section class="panel">
        <h2>Selected GPU Data</h2>
        <div class="tables">
          <div>
            <h3>Observed Telemetry</h3>
            <div id="observedTable"></div>
          </div>
          <div>
            <h3>Predicted Data</h3>
            <div id="predictionTable"></div>
          </div>
        </div>
      </section>
    </div>

    <aside class="chat">
      <div class="chat-header">
        <div>
          <h2 id="chatTitle">Rack Copilot</h2>
          <div id="chatSub" class="subtle">Click a rack to inspect it.</div>
        </div>
        <button id="detailsBtn" class="secondary" disabled>Details</button>
      </div>
      <div id="messages" class="messages"></div>
      <div class="chat-compose">
        <textarea id="operatorNote" placeholder="Tell the agent what the technician sees or hears. Example: Rack 4 rear manifold looks wet and the coolant flow alarm is amber."></textarea>
        <div class="compose-actions">
          <button id="attachBtn" class="secondary icon" title="Attach image or audio">+</button>
          <button id="runBtn">Run Analysis</button>
          <button id="clearBtn" class="secondary">Clear</button>
        </div>
        <input id="imageFile" class="hidden" type="file" accept="image/png,image/jpeg" />
        <input id="audioFile" class="hidden" type="file" accept="audio/wav,audio/mpeg" />
        <div id="attachments" class="attachments"></div>
      </div>
    </aside>
  </div>

  <div id="modalBackdrop" class="modal-backdrop">
    <div class="modal">
      <div class="modal-head">
        <div>
          <h2 id="modalTitle">Rack Details</h2>
          <div id="modalSub" class="subtle"></div>
        </div>
        <button id="closeModal" class="secondary">Close</button>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="forecast">Forecast</button>
        <button class="tab" data-tab="actions">Action Scores</button>
        <button class="tab" data-tab="report">Report</button>
      </div>
      <div id="modalBody" class="modal-body"></div>
    </div>
  </div>

  <script>
    let racks = [];
    let selectedRack = null;
    let selectedGpu = null;
    let workloadDemand = null;
    let activeTab = "forecast";
    let viewMode = "racks";
    let appMode = "simulation";

    const $ = (id) => document.getElementById(id);
    const riskRank = {SAFE: 1, WATCH: 2, HIGH: 3, CRITICAL: 4};

    function setStatus(text) {
      $("status").textContent = text || "";
    }

    function setMode(mode) {
      appMode = mode;
      $("simulationModeBtn").classList.toggle("active", mode === "simulation");
      $("realModeBtn").classList.toggle("active", mode === "real");
      $("realInputPanel").classList.toggle("hidden", mode !== "real");
      $("simulateBtn").classList.toggle("hidden", mode !== "simulation");
      setStatus(mode === "real" ? "Real input mode ready" : "Simulation mode ready");
      renderChat(selectedGpu);
    }

    async function simulateFleet() {
      $("simulateBtn").disabled = true;
      setStatus("Simulating inference demand, scheduler placement, and GPU telemetry...");
      try {
        const response = await fetch("/api/simulate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({rack_count: 3, gpus_per_rack: 8})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Simulation failed");
        racks = data.racks || [];
        workloadDemand = data.workload || null;
        selectedRack = null;
        selectedGpu = null;
        viewMode = "racks";
        renderSummary();
        renderMap();
        renderTables(null);
        renderChat(null);
        $("detailsBtn").disabled = true;
        setStatus("Inference workload simulated at " + new Date(data.generated_at).toLocaleTimeString());
      } catch (err) {
        setStatus(err.message || String(err));
      } finally {
        $("simulateBtn").disabled = false;
      }
    }

    function inputNumber(id, fallback) {
      const value = Number($(id).value);
      return Number.isFinite(value) ? value : fallback;
    }

    function realTelemetryFromForm() {
      const assigned = inputNumber("realAssignedTraffic", 70);
      const headroom = inputNumber("realHeadroom", Math.max(0, 72 - assigned));
      const queue = inputNumber("realQueue", 0);
      const sm = inputNumber("realSmUtil", 70);
      const mem = inputNumber("realMemUtil", 55);
      return {
        rack_id: $("realRackId").value || "rack-1",
        gpu_id: $("realGpuId").value || "gpu-1",
        gpu_temp_c: inputNumber("realGpuTemp", 45),
        ambient_temp_c: 24,
        coolant_temp_c: 28,
        cooling_flow_lpm: inputNumber("realCoolingFlow", 1.1),
        cooling_command_pct: 85,
        fan_speed_rpm: inputNumber("realFanSpeed", 6200),
        fan_command_pct: 90,
        gpu_power_w: inputNumber("realPower", 260),
        psu_voltage_v: 12.0,
        rack_power_kw: 5.2,
        power_spike_ratio: sm >= 80 ? 1.35 : 1.08,
        workload_type: sm >= 80 ? "high_spike_compute" : "steady_inference",
        gpu_util_pct: Math.max(sm, mem),
        sm_util_pct: sm,
        memory_util_pct: mem,
        inference_queue_len: queue,
        network_latency_ms: inputNumber("realLatency", 8),
        packet_loss_pct: 0.1,
        request_rate_rps: Math.max(1, Math.round(assigned * 1.7 + queue * 0.2)),
        active_batches: Math.max(1, Math.round(queue / 16)),
        assigned_traffic_pct: assigned,
        workload_demand_units: Math.round(assigned * 12 + queue * 3),
        prompt_tokens: 1800,
        output_tokens: 620,
        alternative_capacity_pct: headroom,
        latency_tolerance_ms: 45,
        batch_jobs_present: queue > 80
      };
    }

    async function fileToText(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = reject;
        reader.readAsText(file);
      });
    }

    async function analyzeRealInput() {
      $("analyzeRealBtn").disabled = true;
      setStatus("Analyzing real/operator telemetry...");
      try {
        const file = $("realDatasetFile").files[0];
        const datasetText = file ? await fileToText(file) : $("realDataset").value.trim();
        const body = datasetText
          ? {dataset_text: datasetText, use_crusoe: $("useCrusoe").checked}
          : {telemetry: realTelemetryFromForm(), use_crusoe: $("useCrusoe").checked};
        const response = await fetch("/api/real-input", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Real input analysis failed");
        racks = data.racks || [];
        workloadDemand = data.workload || null;
        selectedRack = null;
        selectedGpu = null;
        viewMode = "racks";
        renderSummary();
        renderMap();
        renderTables(null);
        renderChat(null);
        $("detailsBtn").disabled = true;
        setStatus("Real input analyzed at " + new Date(data.generated_at).toLocaleTimeString());
      } catch (err) {
        setStatus(err.message || String(err));
      } finally {
        $("analyzeRealBtn").disabled = false;
      }
    }

    function loadRealExample() {
      $("realDatasetFile").value = "";
      $("realDataset").value = JSON.stringify([
        {"rack_id":"rack-1","gpu_id":"gpu-1","gpu_temp_c":61.8,"assigned_traffic_pct":86,"alternative_capacity_pct":6,"inference_queue_len":112,"gpu_power_w":342,"sm_util_pct":93,"memory_util_pct":66,"cooling_flow_lpm":0.55,"fan_speed_rpm":5200,"network_latency_ms":19},
        {"rack_id":"rack-1","gpu_id":"gpu-2","gpu_temp_c":57.2,"assigned_traffic_pct":78,"alternative_capacity_pct":12,"inference_queue_len":71,"gpu_power_w":303,"sm_util_pct":82,"memory_util_pct":61,"cooling_flow_lpm":0.82,"fan_speed_rpm":6500,"network_latency_ms":12},
        {"rack_id":"rack-1","gpu_id":"gpu-3","gpu_temp_c":43.4,"assigned_traffic_pct":44,"alternative_capacity_pct":28,"inference_queue_len":8,"gpu_power_w":182,"sm_util_pct":46,"memory_util_pct":38,"cooling_flow_lpm":1.26,"fan_speed_rpm":5900,"network_latency_ms":5},
        {"rack_id":"rack-2","gpu_id":"gpu-1","gpu_temp_c":39.8,"assigned_traffic_pct":34,"alternative_capacity_pct":38,"inference_queue_len":4,"gpu_power_w":151,"sm_util_pct":34,"memory_util_pct":35,"cooling_flow_lpm":1.31,"fan_speed_rpm":5700,"network_latency_ms":4}
      ], null, 2);
      setStatus("Example real dataset loaded");
    }

    function clearRealInput() {
      $("realDataset").value = "";
      $("realDatasetFile").value = "";
      setStatus("Real input cleared");
    }

    function renderSummary() {
      const allGpus = racks.flatMap((rack) => rack.gpus || []);
      const critical = allGpus.filter((gpu) => gpu.forecast.risk === "CRITICAL").length;
      const safe = allGpus.filter((gpu) => gpu.forecast.risk === "SAFE").length;
      const totalHeadroom = allGpus.reduce((sum, gpu) => sum + Number(gpu.telemetry.alternative_capacity_pct || 0), 0);
      $("summary").innerHTML = [
        metric("Physical Racks", racks.length),
        metric("Total GPUs", allGpus.length),
        metric("Critical GPUs", critical),
        metric("Safe GPUs", safe),
        metric("Inference Request", workloadDemand ? cleanLabel(workloadDemand.request_type) : "none"),
        metric("Queued Jobs", workloadDemand ? workloadDemand.queued_jobs : "none"),
        metric("Demand Units", workloadDemand ? num(workloadDemand.demand_units, 0) : "none"),
        metric("Safe Headroom", num(totalHeadroom, 0) + "%"),
      ].join("");
    }

    function renderMap() {
      if (viewMode === "gpus" && selectedRack) renderGpuMap();
      else renderRackMap();
    }

    function renderRackMap() {
      $("mapTitle").textContent = "Rack Map";
      $("mapSub").textContent = "Click a rack to inspect its 8 GPUs";
      $("backBtn").classList.add("hidden");
      $("rackGrid").innerHTML = racks.map((rack) => `
        <button class="rack rack-card" data-rack="${escapeHtml(rack.rack_id)}">
          <div class="dual-strip">
            <span class="strip-segment ${rack.rack_temp_state || "SAFE"}"></span>
            <span class="strip-segment ${rack.rack_state}"></span>
          </div>
          <div class="rack-upper">
            <div class="rack-head">
              <span class="rack-id">${escapeHtml(rack.rack_id.toUpperCase())}</span>
              <span class="risk ${rack.rack_temp_state || "SAFE"}">${formatRackState(rack.rack_temp_state || "SAFE")}</span>
            </div>
            <div class="temp">${num(rack.rack_temp_c, 1)} C</div>
            <div class="subtle">Rack current temperature</div>
          </div>
          <div class="rack-lower">
            <div class="rack-head">
              <span class="subtle">GPU state</span>
              <span class="risk ${rack.rack_state}">${formatRackState(rack.rack_state)}</span>
            </div>
            <div class="rack-lines">
              <span>Critical GPUs: ${rack.critical_count}/${rack.gpu_count}</span>
              <span>Watch/High GPUs: ${rack.watch_count + rack.high_count}/${rack.gpu_count}</span>
              <span>Avg assigned traffic: ${num(rack.avg_assigned_traffic_pct, 0)}%</span>
              <span>Safe headroom: ${num(rack.safe_headroom_pct, 0)}%</span>
              <span>Queued jobs: ${rack.queued_jobs || 0}</span>
              <span>Max GPU temp: ${num(rack.max_gpu_temp_c, 1)} C</span>
              <span>Top GPU: ${escapeHtml(rack.top_gpu_id || "none")} (${rack.top_risk})</span>
            </div>
            <div class="gpu-mini-grid">
              ${(rack.gpus || []).map((gpu) => `<span class="gpu-mini ${gpu.forecast.risk}" title="${escapeHtml(gpu.telemetry.gpu_id)} ${gpu.forecast.risk}"></span>`).join("")}
            </div>
          </div>
        </button>
      `).join("");
      document.querySelectorAll(".rack").forEach((card) => {
        card.addEventListener("click", () => openRack(card.dataset.rack));
      });
    }

    function renderGpuMap() {
      $("mapTitle").textContent = selectedRack.rack_id.toUpperCase() + " GPUs";
      $("mapSub").textContent = "Click a GPU to update diagnosis and recommendation";
      $("backBtn").classList.remove("hidden");
      $("rackGrid").innerHTML = (selectedRack.gpus || []).map((gpu) => {
        const t = gpu.telemetry;
        const f = gpu.forecast;
        const d = gpu.diagnosis;
        const selectedClass = selectedGpu && selectedGpu.telemetry.gpu_id === t.gpu_id ? " selected" : "";
        return `
          <button class="rack gpu-card ${f.risk}${selectedClass}" data-gpu="${escapeHtml(t.gpu_id)}">
            <div class="rack-head">
              <span class="rack-id">${escapeHtml(t.gpu_id.toUpperCase())}</span>
              <span class="risk ${f.risk}">${f.risk}</span>
            </div>
            <div class="temp">${num(t.gpu_temp_c, 1)} C</div>
            <div class="rack-lines">
              <span>Peak forecast: ${num(f.peak_temp_c, 1)} C</span>
              <span>Assigned traffic: ${num(t.assigned_traffic_pct, 0)}%</span>
              <span>Safe headroom: ${num(t.alternative_capacity_pct, 0)}%</span>
              <span>Queue: ${t.inference_queue_len || 0} jobs</span>
              <span>Threshold: ${f.time_to_threshold_s === null ? "not in horizon" : secondsLabel(f.time_to_threshold_s)}</span>
              <span>${cleanCause(d.likely_cause)}</span>
            </div>
          </button>
        `;
      }).join("");
      document.querySelectorAll("[data-gpu]").forEach((card) => {
        card.addEventListener("click", () => selectGpu(card.dataset.gpu));
      });
    }

    function openRack(rackId) {
      selectedRack = racks.find((rack) => rack.rack_id === rackId) || null;
      selectedGpu = null;
      viewMode = "gpus";
      renderMap();
      renderTables(null);
      renderChat(null);
      $("detailsBtn").disabled = true;
      setStatus("Opened " + rackId.toUpperCase() + ". Select a GPU.");
    }

    function selectGpu(gpuId) {
      selectedGpu = (selectedRack.gpus || []).find((gpu) => gpu.telemetry.gpu_id === gpuId) || null;
      renderMap();
      renderTables(selectedGpu);
      renderChat(selectedGpu);
      $("detailsBtn").disabled = !selectedGpu;
      setStatus("Selected " + selectedRack.rack_id.toUpperCase() + " / " + gpuId.toUpperCase());
    }

    function backToRacks() {
      selectedRack = null;
      selectedGpu = null;
      viewMode = "racks";
      renderMap();
      renderTables(null);
      renderChat(null);
      $("detailsBtn").disabled = true;
      setStatus("Back to rack map.");
    }

    function renderTables(item) {
      if (!item) {
        $("observedTable").innerHTML = emptyBox("Select a GPU to see observed telemetry.");
        $("predictionTable").innerHTML = emptyBox("Select a GPU to see predicted data.");
        return;
      }
      $("observedTable").innerHTML = tableHtml(["Parameter", "Value", "Unit"], item.observed_table.map((row) => [
        row.parameter, row.value, row.unit
      ]));
      $("predictionTable").innerHTML = tableHtml(["Time", "GPU Temp", "Power"], item.prediction_table.map((row) => [
        secondsLabel(row.t_s), num(row.gpu_temp_c, 1) + " C", num(row.power_w, 1) + " W"
      ]));
    }

    function renderChat(item) {
      if (!item) {
        $("chatTitle").textContent = "GPU Copilot";
        $("chatSub").textContent = "Select a rack, then select one GPU.";
        const workflowText = appMode === "real"
          ? "Fill manual telemetry or paste a dataset, then analyze it. Open a rack and select a GPU for diagnosis and actions."
          : "Click Simulate Inference Workload to generate an inference request, scheduler placement, telemetry, and forecast. Open a rack, then select a GPU for diagnosis and actions.";
        $("messages").innerHTML = `
          <div class="message">
            <div class="message-title">Workflow</div>
            <div>${workflowText}</div>
          </div>
        `;
        return;
      }
      const t = item.telemetry;
      const f = item.forecast;
      const d = item.diagnosis;
      const action = item.recommendation.primary_action;
      const migrationCandidate = (item.recommendation.candidates || []).find((candidate) => candidate.action_id === "migrate_inference_traffic");
      const migrationControl = migrationCandidate && f.risk !== "SAFE"
        ? `<button class="link-btn" id="prepareMigration">Prepare migration plan</button>`
        : "";
      $("chatTitle").textContent = t.rack_id.toUpperCase() + " / " + t.gpu_id.toUpperCase();
      $("chatSub").textContent = "Risk " + f.risk + " | " + cleanCause(item.simulated_case || d.likely_cause);
      const crossing = f.time_to_threshold_s === null ? "No threshold crossing in horizon" : "Threshold in " + secondsLabel(f.time_to_threshold_s);
      $("messages").innerHTML = `
        <div class="message">
          <div class="message-title">Diagnosis</div>
          <div><strong>${cleanCause(d.likely_cause)}</strong> with ${Math.round(d.confidence * 100)}% confidence.</div>
          <div class="chips">
            <span class="chip">Current ${num(t.gpu_temp_c, 1)} C</span>
            <span class="chip">Peak ${num(f.peak_temp_c, 1)} C</span>
            <span class="chip">Assigned ${num(t.assigned_traffic_pct, 0)}%</span>
            <span class="chip">Headroom ${num(t.alternative_capacity_pct, 0)}%</span>
            <span class="chip">${crossing}</span>
          </div>
        </div>
        <div class="message">
          <div class="message-title">Why</div>
          <ul>${(d.reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>
        </div>
        <div class="message">
          <div class="message-title">Recommendation</div>
          <div><strong>${escapeHtml(action.label)}</strong></div>
          <div>${escapeHtml(action.expected_impact)}</div>
          <div class="chips">
            <span class="chip">Score ${action.score}</span>
            <span class="chip">Cost ${escapeHtml(action.operational_cost)}</span>
          </div>
          <div class="link-row">
            <button class="link-btn" id="viewActions">Explain action score</button>
            ${migrationControl}
          </div>
        </div>
      `;
      $("viewActions").addEventListener("click", () => openModal("actions"));
      const migrationButton = $("prepareMigration");
      if (migrationButton) migrationButton.addEventListener("click", prepareMigrationPlan);
    }

    async function runAnalysis() {
      if (!selectedGpu) {
        setStatus("Select a GPU first.");
        return;
      }
      $("runBtn").disabled = true;
      setStatus("Reviewing evidence...");
      try {
        const [imageDataUrl, audioDataUrl] = await Promise.all([
          fileToDataUrl($("imageFile")),
          fileToDataUrl($("audioFile"))
        ]);
        const note = $("operatorNote").value.trim();
        if (note) appendMessage("user", note);
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            telemetry: selectedGpu.telemetry,
            forecast: selectedGpu.forecast,
            text_note: note,
            image_data_url: imageDataUrl,
            audio_data_url: audioDataUrl,
            use_crusoe: $("useCrusoe").checked
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Analysis failed");
        data.simulated_case = selectedGpu.simulated_case;
        data.prediction_source = selectedGpu.prediction_source;
        const idx = selectedRack.gpus.findIndex((gpu) => gpu.telemetry.gpu_id === selectedGpu.telemetry.gpu_id);
        if (idx >= 0) selectedRack.gpus[idx] = data;
        selectedGpu = data;
        recomputeRack(selectedRack);
        renderSummary();
        renderMap();
        renderTables(selectedGpu);
        renderChat(selectedGpu);
        if (data.crusoe && data.crusoe.error) setStatus(data.crusoe.error);
        else setStatus(data.crusoe && data.crusoe.used ? "Nemotron reviewed the evidence." : "Local evidence review complete.");
      } catch (err) {
        setStatus(err.message || String(err));
      } finally {
        $("runBtn").disabled = false;
      }
    }

    async function approveAction() {
      if (!selectedGpu) return;
      setStatus("Simulating mitigation...");
      const response = await fetch("/api/mitigate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({result: selectedGpu})
      });
      const data = await response.json();
      if (!response.ok) {
        setStatus(data.error || "Mitigation failed");
        return;
      }
      appendMessage("assistant", "Mitigation simulation: risk moves from " + data.before.risk + " to " + data.after.risk + ", peak temp changes from " + data.before.peak_temp_c + " C to " + data.after.peak_temp_c + " C.");
      setStatus("Mitigation simulated.");
    }

    async function prepareMigrationPlan() {
      if (!selectedGpu) return;
      const button = $("prepareMigration");
      if (button) button.disabled = true;
      setStatus("Searching for safe target GPU capacity...");
      try {
        const response = await fetch("/api/propose-migration", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({source: selectedGpu, racks})
        });
        const plan = await response.json();
        if (!response.ok) throw new Error(plan.error || "Migration planning failed");
        if (!plan.available) {
          const scale = plan.scale_recommendation || {};
          appendHtmlMessage("warning", `
            <div class="message-title">No Safe Migration Target</div>
            <div>${escapeHtml(plan.reason || "No safe target capacity is available right now.")}</div>
            <div class="chips">
              <span class="chip">Source ${escapeHtml(plan.source.rack_id || "")} / ${escapeHtml(plan.source.gpu_id || "")}</span>
              <span class="chip">Recommended: ${escapeHtml(scale.label || "Add GPU rack capacity")}</span>
              <span class="chip">Extra racks: ${scale.additional_racks_needed || 0}</span>
            </div>
          `);
          setStatus("No safe migration target found.");
          return;
        }
        appendMigrationPlan(plan);
        setStatus("Migration plan ready for approval.");
      } catch (err) {
        setStatus(err.message || String(err));
      } finally {
        if (button) button.disabled = false;
      }
    }

    function appendMigrationPlan(plan) {
      const basis = plan.basis || {};
      const targetRows = (plan.targets || []).map((target) => `
        <tr>
          <td>${escapeHtml(target.rack_id)} / ${escapeHtml(target.gpu_id)}</td>
          <td>${escapeHtml(target.risk)}</td>
          <td>${num(target.current_temp_c, 1)} C</td>
          <td>${num(target.assigned_traffic_pct, 0)}%</td>
          <td>${num(target.available_capacity_pct, 0)}%</td>
          <td>${num(target.traffic_share_pct, 0)}%</td>
        </tr>
      `).join("");
      const guardrails = (plan.guardrails || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
      const html = `
        <div class="message-title">Migration Plan Awaiting Approval</div>
        <div>Move <strong>${num(plan.traffic_percent, 0)}%</strong> of high-compute inference traffic away from <strong>${escapeHtml(plan.source.rack_id)} / ${escapeHtml(plan.source.gpu_id)}</strong>.</div>
        <div class="chips">
          <span class="chip">Risk ${escapeHtml(plan.expected.before.risk)} to ${escapeHtml(plan.expected.after.risk)}</span>
          <span class="chip">Peak ${num(plan.expected.before.peak_temp_c, 1)} C to ${num(plan.expected.after.peak_temp_c, 1)} C</span>
          <span class="chip">Source excess ${num(basis.source_excess_pct, 0)}%</span>
          <span class="chip">Target headroom ${num(basis.target_safe_headroom_pct, 0)}%</span>
        </div>
        <div class="subtle" style="margin-top:8px">${escapeHtml(basis.formula || "Traffic percent is based on source overload and target headroom.")}</div>
        <div style="height:10px"></div>
        <table>
          <thead><tr><th>Target GPU</th><th>Risk</th><th>Current</th><th>Assigned</th><th>Headroom</th><th>Traffic</th></tr></thead>
          <tbody>${targetRows}</tbody>
        </table>
        <ul class="guardrails">${guardrails}</ul>
        <button class="approve-btn">Approve simulated migration</button>
      `;
      const div = appendHtmlMessage("plan", html);
      const approveButton = div.querySelector("button.approve-btn");
      approveButton.addEventListener("click", () => approveMigration(plan, approveButton));
    }

    async function approveMigration(plan, button) {
      if (!selectedGpu) return;
      if (button) button.disabled = true;
      let completed = false;
      setStatus("Engineer approved. Simulating migration...");
      try {
        const response = await fetch("/api/execute-migration", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({source: selectedGpu, plan, racks})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Migration execution failed");
        if (!data.executed) {
          appendMessage("assistant", data.reason || "Migration was not executed.");
          setStatus("Migration not executed.");
          return;
        }
        if (data.updated_result) {
          data.updated_result.last_migration = {
            approved_at: data.approved_at,
            traffic_percent: data.traffic_percent,
            before: data.before,
            after: data.after,
            targets: data.targets
          };
          replaceGpuResult(data.updated_result);
          selectedGpu = data.updated_result;
        }
        (data.target_updates || []).forEach((targetUpdate) => replaceGpuResult(targetUpdate));
        if (data.updated_result || (data.target_updates || []).length) {
          refreshFleetViewAfterMutation();
        }
        appendHtmlMessage("success", `
          <div class="message-title">Migration Approved And Simulated</div>
          <div>The selected GPU forecast has been updated with the approved post-migration prediction.</div>
          <div class="chips">
            <span class="chip">Moved ${num(data.traffic_percent, 0)}%</span>
            <span class="chip">Risk ${escapeHtml(data.before.risk)} to ${escapeHtml(data.after.risk)}</span>
            <span class="chip">Peak ${num(data.before.peak_temp_c, 1)} C to ${num(data.after.peak_temp_c, 1)} C</span>
          </div>
        `);
        completed = true;
        if (button) button.textContent = "Approved";
        setStatus("Post-migration prediction applied.");
      } catch (err) {
        setStatus(err.message || String(err));
      } finally {
        if (button && !completed) button.disabled = false;
      }
    }

    function replaceGpuResult(updatedGpu) {
      if (!updatedGpu || !updatedGpu.telemetry) return;
      const rack = racks.find((item) => item.rack_id === updatedGpu.telemetry.rack_id);
      if (!rack) return;
      const idx = rack.gpus.findIndex((gpu) => gpu.telemetry.gpu_id === updatedGpu.telemetry.gpu_id);
      if (idx >= 0) rack.gpus[idx] = updatedGpu;
      recomputeRack(rack);
      if (selectedRack && selectedRack.rack_id === rack.rack_id) selectedRack = rack;
    }

    function refreshFleetViewAfterMutation() {
      if (selectedRack) recomputeRack(selectedRack);
      renderSummary();
      renderMap();
      renderTables(selectedGpu);
      $("chatTitle").textContent = selectedGpu.telemetry.rack_id.toUpperCase() + " / " + selectedGpu.telemetry.gpu_id.toUpperCase();
      $("chatSub").textContent = "Risk " + selectedGpu.forecast.risk + " | post-migration forecast";
      if ($("modalBackdrop").classList.contains("open")) {
        $("modalTitle").textContent = selectedGpu.telemetry.rack_id.toUpperCase() + " / " + selectedGpu.telemetry.gpu_id.toUpperCase();
        $("modalSub").textContent = selectedGpu.forecast.risk + " | post-migration forecast";
        renderModalBody();
      }
    }

    function openModal(tabName) {
      if (!selectedGpu) return;
      activeTab = tabName || activeTab;
      $("modalBackdrop").classList.add("open");
      $("modalTitle").textContent = selectedGpu.telemetry.rack_id.toUpperCase() + " / " + selectedGpu.telemetry.gpu_id.toUpperCase();
      $("modalSub").textContent = selectedGpu.forecast.risk + " | " + cleanCause(selectedGpu.diagnosis.likely_cause);
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === activeTab));
      renderModalBody();
    }

    function renderModalBody() {
      if (!selectedGpu) return;
      if (activeTab === "forecast") renderForecastTab();
      if (activeTab === "actions") renderActionsTab();
      if (activeTab === "report") renderReportTab();
    }

    function renderForecastTab() {
      const f = selectedGpu.forecast;
      $("modalBody").innerHTML = `
        <div class="cards">
          ${infoCard("Current GPU", num(f.current_temp_c, 1) + " C")}
          ${infoCard("Forecast Peak", num(f.peak_temp_c, 1) + " C")}
          ${infoCard("Time To Threshold", f.time_to_threshold_s === null ? "None" : secondsLabel(f.time_to_threshold_s))}
          ${infoCard("Prediction Source", predictionSourceLabel(selectedGpu.prediction_source))}
        </div>
        <div style="height:14px"></div>
        <canvas id="chart" width="1200" height="520"></canvas>
      `;
      drawChart(f.trajectory || [], f.threshold_c);
    }

    function renderActionsTab() {
      const t = selectedGpu.telemetry;
      const candidates = selectedGpu.recommendation.candidates || [];
      $("modalBody").innerHTML = `
        <div class="message">
          <div class="message-title">How the score works</div>
          <div>Score = expected risk reduction minus latency penalty, migration cost, SLA penalty, operator preference penalty, and confidence penalty. Higher score means better one-tap choice for this GPU.</div>
        </div>
        <div style="height:12px"></div>
        <div class="cards">
          ${candidates.map((c) => actionCard(c, t)).join("")}
        </div>
      `;
      document.querySelectorAll(".action-card").forEach((card) => {
        card.addEventListener("click", () => card.classList.toggle("open"));
      });
    }

    function renderReportTab() {
      const report = selectedGpu.report || {};
      const evidence = Array.isArray(report.evidence_used) ? report.evidence_used : [];
      const criticalTone = selectedGpu.forecast.risk === "CRITICAL" ? "alert" : "";
      $("modalBody").innerHTML = `
        <div class="report-list">
          ${reportBlock("Incident summary", report.incident_summary || "No summary available.", criticalTone)}
          ${reportBlock("Likely cause", cleanCause(report.likely_cause || selectedGpu.diagnosis.likely_cause), criticalTone)}
          ${reportBlock("Evidence used", evidence.length ? "<ul>" + evidence.map((x) => "<li>" + escapeHtml(x) + "</li>").join("") + "</ul>" : "No evidence listed.", criticalTone)}
          ${reportBlock("Operator next step", report.operator_next_step || selectedGpu.recommendation.primary_action.label)}
          ${reportBlock("Escalation note", report.escalation_note || "No escalation needed.")}
        </div>
      `;
    }

    function actionCard(candidate, telemetry) {
      return `
        <div class="action-card">
          <div style="display:flex;justify-content:space-between;gap:12px">
            <strong>${escapeHtml(candidate.label)}</strong>
            <span class="action-score">${candidate.score}</span>
          </div>
          <div class="subtle">${escapeHtml(candidate.rationale)}</div>
          <div class="hidden-detail" style="margin-top:10px">
            <table>
              <tr><th>Input</th><th>Value</th></tr>
              <tr><td>GPU risk</td><td>${selectedGpu.forecast.risk}</td></tr>
              <tr><td>Diagnosis confidence</td><td>${Math.round(selectedGpu.diagnosis.confidence * 100)}%</td></tr>
              <tr><td>Safe traffic headroom</td><td>${num(telemetry.alternative_capacity_pct, 0)}%</td></tr>
              <tr><td>Assigned inference traffic</td><td>${num(telemetry.assigned_traffic_pct, 0)}%</td></tr>
              <tr><td>Latency tolerance</td><td>${num(telemetry.latency_tolerance_ms, 0)} ms</td></tr>
              <tr><td>Expected impact</td><td>${escapeHtml(candidate.expected_impact)}</td></tr>
            </table>
          </div>
        </div>
      `;
    }

    function recomputeRack(rack) {
      const risks = rack.gpus.map((gpu) => gpu.forecast.risk);
      rack.critical_count = risks.filter((risk) => risk === "CRITICAL").length;
      rack.high_count = risks.filter((risk) => risk === "HIGH").length;
      rack.watch_count = risks.filter((risk) => risk === "WATCH").length;
      rack.safe_count = risks.filter((risk) => risk === "SAFE").length;
      const temps = rack.gpus.map((gpu) => Number(gpu.telemetry.gpu_temp_c)).filter((value) => !Number.isNaN(value));
      rack.max_gpu_temp_c = temps.length ? Math.max(...temps) : null;
      rack.avg_gpu_temp_c = temps.length ? temps.reduce((a, b) => a + b, 0) / temps.length : null;
      rack.rack_temp_c = rack.avg_gpu_temp_c;
      rack.rack_temp_state = rack.rack_temp_c > 80 ? "CRITICAL" : (rack.rack_temp_c >= 50 ? "MEDIUM" : "SAFE");
      const loads = rack.gpus.map((gpu) => Number(gpu.telemetry.assigned_traffic_pct || 0));
      const headrooms = rack.gpus.map((gpu) => Number(gpu.telemetry.alternative_capacity_pct || 0));
      rack.avg_assigned_traffic_pct = loads.length ? loads.reduce((a, b) => a + b, 0) / loads.length : 0;
      rack.safe_headroom_pct = headrooms.reduce((a, b) => a + b, 0);
      rack.queued_jobs = rack.gpus.reduce((sum, gpu) => sum + Number(gpu.telemetry.inference_queue_len || 0), 0);
      const halfGpuCount = rack.gpus.length / 2;
      rack.rack_state = rack.critical_count > halfGpuCount ? "CRITICAL" : (rack.critical_count === halfGpuCount ? "MEDIUM" : "SAFE");
      const top = [...rack.gpus].sort((a, b) => (riskRank[b.forecast.risk] - riskRank[a.forecast.risk]) || (b.forecast.peak_temp_c - a.forecast.peak_temp_c))[0];
      rack.top_gpu_id = top.telemetry.gpu_id;
      rack.top_risk = top.forecast.risk;
      rack.dominant_cause = top.diagnosis.likely_cause;
    }

    function appendMessage(kind, text) {
      const div = document.createElement("div");
      div.className = "message " + (kind === "user" ? "user" : "");
      div.textContent = text;
      $("messages").appendChild(div);
      $("messages").scrollTop = $("messages").scrollHeight;
    }

    function appendHtmlMessage(tone, html) {
      const div = document.createElement("div");
      div.className = "message " + (tone || "");
      div.innerHTML = html;
      $("messages").appendChild(div);
      $("messages").scrollTop = $("messages").scrollHeight;
      return div;
    }

    function fileToDataUrl(input) {
      const file = input.files && input.files[0];
      if (!file) return Promise.resolve("");
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    function drawChart(points, threshold) {
      const canvas = $("chart");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(700, Math.floor(rect.width * scale));
      canvas.height = Math.max(280, Math.floor(rect.height * scale));
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      if (!points.length) return;
      const pad = {l: 50, r: 18, t: 20, b: 36};
      const xs = points.map((p) => p.t_s);
      const ys = points.map((p) => p.gpu_temp_c).concat([threshold]);
      const maxX = Math.max(...xs);
      const minY = Math.floor(Math.min(...ys) - 4);
      const maxY = Math.ceil(Math.max(...ys) + 4);
      const x = (v) => pad.l + (v / maxX) * (w - pad.l - pad.r);
      const y = (v) => h - pad.b - ((v - minY) / (maxY - minY)) * (h - pad.t - pad.b);

      ctx.strokeStyle = "#e4e7df";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const yy = pad.t + i * (h - pad.t - pad.b) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.l, yy);
        ctx.lineTo(w - pad.r, yy);
        ctx.stroke();
      }
      ctx.strokeStyle = "#b42318";
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(pad.l, y(threshold));
      ctx.lineTo(w - pad.r, y(threshold));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = "#087f8c";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      points.forEach((p, i) => {
        if (i === 0) ctx.moveTo(x(p.t_s), y(p.gpu_temp_c));
        else ctx.lineTo(x(p.t_s), y(p.gpu_temp_c));
      });
      ctx.stroke();
      ctx.fillStyle = "#66716b";
      ctx.font = "12px system-ui";
      ctx.fillText(maxY + " C", 8, pad.t + 4);
      ctx.fillText(minY + " C", 8, h - pad.b);
      ctx.fillText("0", pad.l, h - 12);
      ctx.fillText(Math.round(maxX / 60) + " min", w - pad.r - 42, h - 12);
      ctx.fillStyle = "#b42318";
      ctx.fillText("threshold", w - pad.r - 72, y(threshold) - 8);
    }

    function metric(label, value) {
      return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`;
    }

    function infoCard(label, value) {
      return `<div class="info-card"><div class="subtle">${label}</div><strong>${value}</strong></div>`;
    }

    function reportBlock(label, value, tone = "") {
      return `<div class="report-block ${tone}"><strong>${label}</strong><div>${value}</div></div>`;
    }

    function emptyBox(text) {
      return `<div class="message subtle">${escapeHtml(text)}</div>`;
    }

    function tableHtml(headers, rows) {
      return `<table><thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    }

    function secondsLabel(value) {
      if (value === null || value === undefined) return "none";
      const min = Number(value) / 60;
      return min >= 1 ? min.toFixed(1) + " min" : Math.round(Number(value)) + " s";
    }

    function num(value, digits) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "missing";
      return Number(value).toFixed(digits);
    }

    function cleanCause(value) {
      return escapeHtml(String(value || "").replaceAll("_", " "));
    }

    function cleanLabel(value) {
      return escapeHtml(String(value || "").replaceAll("_", " "));
    }

    function predictionSourceLabel(value) {
      const labels = {
        node_checkpoint: "Neural ODE",
        simulator_forecast_placeholder: "Simulator placeholder",
        workload_driven_thermal_simulation: "Workload-driven simulation",
        post_migration_simulation: "Post-migration simulation",
        post_migration_target_load: "Post-migration target load",
      };
      return labels[value] || cleanLabel(value || "simulation");
    }

    function formatRackState(value) {
      const labels = {CRITICAL: "Critical", MEDIUM: "Medium", SAFE: "Safe"};
      return labels[value] || value;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    $("simulateBtn").addEventListener("click", simulateFleet);
    $("simulationModeBtn").addEventListener("click", () => setMode("simulation"));
    $("realModeBtn").addEventListener("click", () => setMode("real"));
    $("analyzeRealBtn").addEventListener("click", analyzeRealInput);
    $("loadRealDemoBtn").addEventListener("click", loadRealExample);
    $("clearRealBtn").addEventListener("click", clearRealInput);
    $("backBtn").addEventListener("click", backToRacks);
    $("detailsBtn").addEventListener("click", () => openModal("forecast"));
    $("runBtn").addEventListener("click", runAnalysis);
    $("clearBtn").addEventListener("click", () => {
      $("operatorNote").value = "";
      $("imageFile").value = "";
      $("audioFile").value = "";
      $("attachments").textContent = "";
      setStatus("Cleared operator input.");
    });
    $("attachBtn").addEventListener("click", () => {
      const imageFirst = !$("imageFile").files.length;
      if (imageFirst) $("imageFile").click();
      else $("audioFile").click();
    });
    $("imageFile").addEventListener("change", updateAttachments);
    $("audioFile").addEventListener("change", updateAttachments);
    function updateAttachments() {
      const names = [];
      if ($("imageFile").files[0]) names.push("image: " + $("imageFile").files[0].name);
      if ($("audioFile").files[0]) names.push("audio: " + $("audioFile").files[0].name);
      $("attachments").textContent = names.join(" | ");
    }
    $("closeModal").addEventListener("click", () => $("modalBackdrop").classList.remove("open"));
    $("modalBackdrop").addEventListener("click", (event) => {
      if (event.target === $("modalBackdrop")) $("modalBackdrop").classList.remove("open");
    });
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => openModal(tab.dataset.tab));
    });
    window.addEventListener("resize", () => {
      if ($("modalBackdrop").classList.contains("open") && activeTab === "forecast" && selectedGpu) {
        drawChart(selectedGpu.forecast.trajectory || [], selectedGpu.forecast.threshold_c);
      }
    });

    simulateFleet();
  </script>
</body>
</html>
"""


FIELD_ALIASES = {
    "rack": "rack_id",
    "rackid": "rack_id",
    "rack_id": "rack_id",
    "gpu": "gpu_id",
    "gpuid": "gpu_id",
    "gpu_id": "gpu_id",
    "temperature": "gpu_temp_c",
    "temp": "gpu_temp_c",
    "gpu_temp": "gpu_temp_c",
    "gpu_temp_c": "gpu_temp_c",
    "power": "gpu_power_w",
    "gpu_power": "gpu_power_w",
    "gpu_power_w": "gpu_power_w",
    "queue": "inference_queue_len",
    "queue_len": "inference_queue_len",
    "inference_queue_len": "inference_queue_len",
    "latency": "network_latency_ms",
    "network_latency": "network_latency_ms",
    "network_latency_ms": "network_latency_ms",
    "traffic": "assigned_traffic_pct",
    "assigned_traffic": "assigned_traffic_pct",
    "assigned_traffic_pct": "assigned_traffic_pct",
    "headroom": "alternative_capacity_pct",
    "safe_headroom": "alternative_capacity_pct",
    "alternative_capacity_pct": "alternative_capacity_pct",
    "sm": "sm_util_pct",
    "sm_util": "sm_util_pct",
    "sm_util_pct": "sm_util_pct",
    "memory": "memory_util_pct",
    "memory_util": "memory_util_pct",
    "memory_util_pct": "memory_util_pct",
    "util": "gpu_util_pct",
    "gpu_util": "gpu_util_pct",
    "gpu_util_pct": "gpu_util_pct",
    "cooling_flow": "cooling_flow_lpm",
    "cooling_flow_lpm": "cooling_flow_lpm",
    "fan_speed": "fan_speed_rpm",
    "fan_speed_rpm": "fan_speed_rpm",
    "ambient": "ambient_temp_c",
    "ambient_temp_c": "ambient_temp_c",
    "coolant": "coolant_temp_c",
    "coolant_temp_c": "coolant_temp_c",
}


def _coerce_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if number.is_integer() else number
    return value


def _normalize_telemetry_row(row: dict[str, Any], index: int = 0) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in (row or {}).items():
        alias = FIELD_ALIASES.get(str(key).strip().lower(), str(key).strip())
        normalized[alias] = _coerce_value(value)

    normalized.setdefault("rack_id", "rack-1")
    normalized.setdefault("gpu_id", f"gpu-{index + 1}")
    assigned = float(normalized.get("assigned_traffic_pct") or normalized.get("gpu_util_pct") or 45.0)
    queue = float(normalized.get("inference_queue_len") or 0.0)
    sm = float(normalized.get("sm_util_pct") or max(25.0, min(99.0, assigned + 6.0)))
    mem = float(normalized.get("memory_util_pct") or max(20.0, min(95.0, assigned * 0.72)))

    normalized.setdefault("gpu_temp_c", 28.0 + assigned * 0.32 + min(7.0, queue * 0.03))
    normalized.setdefault("ambient_temp_c", 24.0)
    normalized.setdefault("coolant_temp_c", 28.0)
    normalized.setdefault("gpu_power_w", 65.0 + assigned * 3.2 + queue * 0.12)
    normalized.setdefault("rack_power_kw", 5.0)
    normalized.setdefault("fan_command_pct", 85.0)
    normalized.setdefault("fan_speed_rpm", 6000.0)
    normalized.setdefault("cooling_command_pct", 85.0)
    normalized.setdefault("cooling_flow_lpm", 1.15)
    normalized.setdefault("psu_voltage_v", 12.0)
    normalized.setdefault("power_spike_ratio", 1.35 if sm >= 80 else 1.08)
    normalized.setdefault("workload_type", "high_spike_compute" if sm >= 80 else "steady_inference")
    normalized.setdefault("gpu_util_pct", max(sm, mem, assigned))
    normalized.setdefault("sm_util_pct", sm)
    normalized.setdefault("memory_util_pct", mem)
    normalized.setdefault("network_latency_ms", 6.0)
    normalized.setdefault("packet_loss_pct", 0.05)
    normalized.setdefault("request_rate_rps", max(1.0, assigned * 1.6 + queue * 0.15))
    normalized.setdefault("active_batches", max(1.0, math.ceil(queue / 16.0)))
    normalized.setdefault("workload_demand_units", assigned * 12.0 + queue * 3.0)
    normalized.setdefault("prompt_tokens", 1500)
    normalized.setdefault("output_tokens", 550)
    normalized.setdefault("alternative_capacity_pct", max(0.0, 72.0 - assigned))
    normalized.setdefault("latency_tolerance_ms", 45.0)
    normalized.setdefault("batch_jobs_present", queue > 80)
    return normalized


def _forecast_from_real_telemetry(telemetry: RackTelemetry) -> ForecastSummary:
    current = float(telemetry.gpu_temp_c or 35.0)
    threshold = DEFAULT_THRESHOLD_C
    assigned = float(telemetry.assigned_traffic_pct or telemetry.gpu_util_pct or 45.0)
    queue = float(telemetry.inference_queue_len or 0.0)
    power = float(telemetry.gpu_power_w or (65.0 + assigned * 3.2))
    cooling_flow = float(telemetry.cooling_flow_lpm or 1.15)
    fan_speed = float(telemetry.fan_speed_rpm or 6000.0)
    latency = float(telemetry.network_latency_ms or 5.0)

    cooling_penalty = max(0.0, 1.0 - cooling_flow) * 9.0 + max(0.0, 5200.0 - fan_speed) / 900.0
    load_penalty = max(0.0, assigned - 72.0) * 0.55
    queue_penalty = min(10.0, queue * 0.045)
    power_penalty = max(0.0, power - 260.0) * 0.026
    latency_penalty = max(0.0, latency - 12.0) * 0.22
    peak = max(current + 1.0, current + 2.0 + load_penalty + queue_penalty + power_penalty + cooling_penalty + latency_penalty)

    horizon = 600.0
    trajectory: list[dict[str, float]] = []
    crossing = None
    for step in range(0, 11):
        t_s = step * 60.0
        progress = 1.0 - math.exp(-t_s / 210.0)
        temp = current + (peak - current) * progress
        point_power = power * (1.0 + 0.03 * progress)
        if crossing is None and temp >= threshold:
            crossing = t_s
        trajectory.append({
            "t_s": round(t_s, 2),
            "power_w": round(point_power, 3),
            "gpu_temp_c": round(temp, 3),
            "heatsink_temp_c": round(temp - 3.0, 3),
        })

    convergence = sum(point["gpu_temp_c"] for point in trajectory[-3:]) / 3.0
    risk = classify_risk(current, peak, convergence, crossing, threshold)
    confidence = 0.76 if telemetry.missing_fields else 0.84
    return ForecastSummary(
        threshold_c=threshold,
        horizon_s=horizon,
        current_temp_c=current,
        peak_temp_c=round(peak, 3),
        convergence_temp_c=round(convergence, 3),
        time_to_threshold_s=crossing,
        risk=risk,
        confidence=confidence,
        trajectory=trajectory,
    )


def _parse_dataset_text(text: str) -> list[dict[str, Any]]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "racks" in parsed:
            return parsed["racks"]
        if isinstance(parsed, dict) and "telemetry" in parsed:
            return [parsed["telemetry"]]
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        raise ValueError("JSON dataset must be an object or list")

    reader = csv.DictReader(io.StringIO(stripped))
    return [dict(row) for row in reader]


def _evaluate_real_rows(rows: list[dict[str, Any]], use_crusoe: bool = False) -> dict[str, Any]:
    if rows and isinstance(rows[0], dict) and "gpus" in rows[0]:
        racks = rows
        fleet = [gpu for rack in racks for gpu in rack.get("gpus", [])]
        return _real_response(racks, fleet)

    fleet: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        telemetry_data = _normalize_telemetry_row(row, index)
        telemetry = RackTelemetry.from_dict(telemetry_data)
        forecast = _forecast_from_real_telemetry(telemetry)
        result = evaluate_rack(telemetry, forecast, use_crusoe=use_crusoe)
        result["observed_table"] = telemetry_table(result["telemetry"])
        result["prediction_table"] = prediction_table(ForecastSummary(**result["forecast"]))
        fleet.append(result)

    if not fleet:
        raise ValueError("real input did not contain telemetry rows")

    by_rack: dict[str, list[dict[str, Any]]] = {}
    for gpu in fleet:
        by_rack.setdefault(gpu["telemetry"].get("rack_id", "rack-1"), []).append(gpu)
    racks = [_aggregate_real_rack(rack_id, gpus) for rack_id, gpus in sorted(by_rack.items())]
    return _real_response(racks, fleet)


def _aggregate_real_rack(rack_id: str, gpus: list[dict[str, Any]]) -> dict[str, Any]:
    risks = [gpu["forecast"]["risk"] for gpu in gpus]
    temps = [float(gpu["telemetry"].get("gpu_temp_c", 0.0)) for gpu in gpus]
    rack_temp = sum(temps) / len(temps)
    critical_count = risks.count("CRITICAL")
    high_count = risks.count("HIGH")
    watch_count = risks.count("WATCH")
    safe_count = risks.count("SAFE")
    half_gpu_count = len(gpus) / 2.0
    rack_state = "CRITICAL" if critical_count > half_gpu_count else ("MEDIUM" if critical_count == half_gpu_count else "SAFE")
    rack_temp_state = "CRITICAL" if rack_temp > 80.0 else ("MEDIUM" if rack_temp >= 50.0 else "SAFE")
    top = sorted(gpus, key=lambda gpu: (
        {"SAFE": 1, "WATCH": 2, "HIGH": 3, "CRITICAL": 4}.get(gpu["forecast"]["risk"], 0),
        float(gpu["forecast"].get("peak_temp_c", 0.0)),
    ), reverse=True)[0]
    avg_assigned = sum(float(gpu["telemetry"].get("assigned_traffic_pct", 0.0) or 0.0) for gpu in gpus) / len(gpus)
    headroom = sum(float(gpu["telemetry"].get("alternative_capacity_pct", 0.0) or 0.0) for gpu in gpus)
    queued = sum(float(gpu["telemetry"].get("inference_queue_len", 0.0) or 0.0) for gpu in gpus)
    return {
        "rack_id": rack_id,
        "gpu_count": len(gpus),
        "critical_count": critical_count,
        "high_count": high_count,
        "watch_count": watch_count,
        "safe_count": safe_count,
        "rack_state": rack_state,
        "rack_temp_state": rack_temp_state,
        "rack_temp_c": round(rack_temp, 2),
        "avg_gpu_temp_c": round(rack_temp, 2),
        "max_gpu_temp_c": round(max(temps), 2),
        "avg_assigned_traffic_pct": round(avg_assigned, 2),
        "safe_headroom_pct": round(headroom, 2),
        "queued_jobs": int(round(queued)),
        "top_gpu_id": top["telemetry"].get("gpu_id"),
        "top_risk": top["forecast"].get("risk"),
        "dominant_cause": top["diagnosis"].get("likely_cause"),
        "gpus": gpus,
    }


def _real_response(racks: list[dict[str, Any]], fleet: list[dict[str, Any]]) -> dict[str, Any]:
    total_queue = sum(float(gpu["telemetry"].get("inference_queue_len", 0.0) or 0.0) for gpu in fleet)
    total_demand = sum(float(gpu["telemetry"].get("workload_demand_units", 0.0) or 0.0) for gpu in fleet)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "real_input",
        "workload": {
            "request_type": "real_operator_input",
            "queued_jobs": int(round(total_queue)),
            "demand_units": round(total_demand, 2),
        },
        "racks": racks,
        "fleet": fleet,
    }


class RackGuardianHandler(BaseHTTPRequestHandler):
    server_version = "RackGuardianDemo/0.2"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
            return
        if self.path == "/api/health":
            self._send_json({
                "ok": True,
                "crusoe_key_present": bool(os.getenv("CRUSOE_API_KEY")),
                "simulator": "rack_guardians.simulator",
            })
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            body = self._read_json()
            if self.path == "/api/simulate":
                self._send_json(self._handle_simulate(body))
                return
            if self.path == "/api/analyze":
                self._send_json(self._handle_analyze(body))
                return
            if self.path == "/api/real-input":
                self._send_json(self._handle_real_input(body))
                return
            if self.path == "/api/propose-migration":
                self._send_json(propose_migration_plan(body["source"], body.get("racks", [])))
                return
            if self.path == "/api/execute-migration":
                self._send_json(execute_migration_plan(body["source"], body["plan"], body.get("racks")))
                return
            if self.path == "/api/mitigate":
                self._send_json(simulate_mitigation(body["result"], body.get("action_id")))
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _handle_simulate(self, body: dict[str, Any]) -> dict[str, Any]:
        rack_count = int(body.get("rack_count", 3))
        gpus_per_rack = int(body.get("gpus_per_rack", 8))
        seed = body.get("seed")
        payload = build_inference_fleet(
            rack_count=max(1, min(rack_count, 8)),
            gpus_per_rack=max(1, min(gpus_per_rack, 16)),
            seed=seed,
        )
        racks = payload["racks"]
        fleet = [gpu for rack in racks for gpu in rack["gpus"]]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workload": payload["workload"],
            "racks": racks,
            "fleet": fleet,
        }

    def _handle_analyze(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("telemetry") and body.get("forecast"):
            telemetry = RackTelemetry.from_dict(body["telemetry"])
            forecast = ForecastSummary(**body["forecast"])
        else:
            telemetry, forecast = build_demo_case(body.get("scenario", "cooling_degradation"))

        result = evaluate_rack(
            telemetry,
            forecast,
            text_note=body.get("text_note", ""),
            image_data_url=body.get("image_data_url", ""),
            audio_data_url=body.get("audio_data_url", ""),
            use_crusoe=bool(body.get("use_crusoe", True)),
        )
        result["observed_table"] = telemetry_table(result["telemetry"])
        result["prediction_table"] = prediction_table(ForecastSummary(**result["forecast"]))
        return result

    def _handle_real_input(self, body: dict[str, Any]) -> dict[str, Any]:
        use_crusoe = bool(body.get("use_crusoe", False))
        if body.get("dataset_text"):
            rows = _parse_dataset_text(str(body["dataset_text"]))
        elif body.get("racks"):
            rows = body["racks"]
        elif body.get("telemetry"):
            rows = [body["telemetry"]]
        else:
            raise ValueError("real input requires telemetry, racks, or dataset_text")
        return _evaluate_real_rows(rows, use_crusoe=use_crusoe)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 25 * 1024 * 1024:
            raise ValueError("request body too large")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def run(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), RackGuardianHandler)
    print(f"Rack Guardian demo running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
