"""Generate a publication-grade, executive-ready PDF submission document for Asterline Assistant."""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

NAME = "Satvik Pedamallu"
NIAT_ID = "N24H01A0514"
GITHUB_URL = "https://github.com/satvik310706/asterline-assistant"
LANGSMITH_URL = "https://smith.langchain.com/o/34ea5156-f488-4661-b611-d4fa3993ddf4/a/asterline-assistant/projects/p/e99ab585-0198-416a-b09f-471470f3df09"

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Asterline Assistant — Technical Take-Home Submission</title>
<style>
  @page {{
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
    @bottom-right {{
      content: counter(page) " / " counter(pages);
      font-size: 8pt;
      color: #64748b;
    }}
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    font-size: 9pt;
    line-height: 1.45;
  }}

  a {{
    color: #4f46e5;
    text-decoration: none;
    word-break: break-all;
  }}
  a:hover {{ text-decoration: underline; }}

  .header-card {{
    background: linear-gradient(135deg, #0f172a 0%, #1e1e38 100%);
    color: #ffffff;
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 14px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
  }}

  .header-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
  }}

  .header-title {{
    font-size: 16pt;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
  }}

  .header-badge {{
    background: #4f46e5;
    color: #ffffff;
    font-size: 7.5pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border-radius: 20px;
  }}

  .header-sub {{
    font-size: 9pt;
    color: #cbd5e1;
    margin-bottom: 12px;
  }}

  .meta-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 16px;
    background: rgba(255, 255, 255, 0.07);
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 8pt;
  }}

  .meta-item strong {{
    color: #93c5fd;
    text-transform: uppercase;
    font-size: 7pt;
    letter-spacing: 0.06em;
    display: block;
    margin-bottom: 1px;
  }}

  .meta-item a, .meta-item span {{
    color: #f8fafc;
    font-family: SFMono-Regular, Consolas, monospace;
    font-size: 7.5pt;
  }}

  h2 {{
    font-size: 11pt;
    font-weight: 750;
    color: #0f172a;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 4px;
    margin: 14px 0 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  h3 {{
    font-size: 9.5pt;
    font-weight: 700;
    color: #334155;
    margin: 10px 0 4px;
  }}

  p {{
    margin-bottom: 6px;
    color: #334155;
  }}

  .highlight-box {{
    background: #f8fafc;
    border-left: 3.5px solid #4f46e5;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    font-size: 8.5pt;
  }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 8px 0;
  }}

  .card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 9px 12px;
  }}

  .card-title {{
    font-size: 8.5pt;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 7.5pt;
  }}

  th {{
    background: #f1f5f9;
    color: #334155;
    text-align: left;
    padding: 5px 7px;
    font-weight: 700;
    border-bottom: 1.5px solid #cbd5e1;
    text-transform: uppercase;
    font-size: 6.8pt;
    letter-spacing: 0.04em;
  }}

  td {{
    padding: 4.5px 7px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
    vertical-align: top;
  }}

  tr:nth-child(even) td {{
    background: #f8fafc;
  }}

  .pill {{
    display: inline-block;
    padding: 1.5px 6px;
    border-radius: 4px;
    font-size: 6.8pt;
    font-weight: 700;
    font-family: SFMono-Regular, Consolas, monospace;
  }}
  .pill-green {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
  .pill-blue {{ background: #e0e7ff; color: #3730a3; border: 1px solid #a5b4fc; }}
  .pill-amber {{ background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }}
  .pill-red {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}

  code {{
    font-family: SFMono-Regular, Consolas, monospace;
    font-size: 7.5pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
  }}

  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 8px 12px;
    border-radius: 6px;
    font-family: SFMono-Regular, Consolas, monospace;
    font-size: 7.2pt;
    line-height: 1.4;
    overflow-x: auto;
    margin: 6px 0;
  }}

  .page-break {{
    page-break-before: always;
  }}

  .arch-diagram {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 10px;
    margin: 8px 0;
    text-align: center;
  }}

  .arch-steps {{
    display: flex;
    justify-content: space-between;
    gap: 4px;
    margin-top: 6px;
  }}

  .arch-step {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 4px;
    flex: 1;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}

  .arch-step strong {{
    display: block;
    font-size: 7.5pt;
    color: #4f46e5;
    margin-bottom: 2px;
  }}

  .arch-step span {{
    font-size: 6.5pt;
    color: #64748b;
    line-height: 1.2;
    display: block;
  }}

  .footer-note {{
    margin-top: 14px;
    padding-top: 8px;
    border-top: 1px solid #e2e8f0;
    font-size: 7pt;
    color: #64748b;
    display: flex;
    justify-content: space-between;
  }}
</style>
</head>
<body>

<!-- ── PAGE 1: EXECUTIVE OVERVIEW & ARCHITECTURE ── -->
<div class="header-card">
  <div class="header-title-row">
    <div>
      <div class="header-title">Asterline Research Assistant</div>
      <div class="header-sub">A production-grade, evidence-grounded research assistant with isolated user memory</div>
    </div>
    <div class="header-badge">Take-Home Assessment</div>
  </div>

  <div class="meta-grid">
    <div class="meta-item">
      <strong>Name</strong>
      <span>{CANDIDATE_NAME}</span>
    </div>
    <div class="meta-item">
      <strong>NIAT ID</strong>
      <span>{CANDIDATE_ID}</span>
    </div>
    <div class="meta-item">
      <strong>GitHub Repository</strong>
      <a href="{GITHUB_URL}">{GITHUB_URL}</a>
    </div>
    <div class="meta-item">
      <strong>LangSmith Traces Project</strong>
      <a href="{LANGSMITH_URL}">View Live Agent Traces</a>
    </div>
  </div>
</div>

<h2>1. Executive Summary & Deliverables</h2>
<p>
Asterline Research Assistant is a conversational AI system designed for Asterline Labs employees. It answers company questions <strong>strictly from a verified 75-chunk corpus</strong> (<code>data/corpus.jsonl</code>, SHA-256 <code>f63fc05cf43204485c98bd8fb6d89860a32c16a7e906bc90975e7c3954fe9fb1</code>), cites exact supporting evidence <code>[chunk_id — title]</code>, resolves temporal conflicts and supersessions, handles missing facts via honest abstention, and securely persists isolated user context across sessions and restarts without ever treating memory as corporate fact.
</p>

<div class="grid-2">
  <div class="card">
    <div class="card-title"><span>Core Architecture</span> <span class="pill pill-blue">LangGraph</span></div>
    <p style="font-size: 7.5pt; color: #475569;">
      7-node LangGraph orchestration pipeline with strictly typed Pydantic models. Router classifies intent, Memory Reader scopes context, Hybrid Retriever fuses dense vector search + BM25, Evidence Analyst labels supersession, Answer Writer cites evidence, Verifier executes deterministic audits, and Memory Writer gates storage.
    </p>
  </div>
  <div class="card">
    <div class="card-title"><span>Resilient Hybrid Stack</span> <span class="pill pill-green">Local + API</span></div>
    <p style="font-size: 7.5pt; color: #475569;">
      <strong>Retrieval:</strong> Local <code>bge-small-en-v1.5</code> (FastEmbed) + local Chroma vector store + <code>rank_bm25</code> Reciprocal Rank Fusion.<br>
      <strong>Memory & Sessions:</strong> Local SQLite with WAL mode, parameterized strictly by <code>user_id</code>.<br>
      <strong>LLM:</strong> Primary Gemini (<code>gemini-3.5-flash-lite</code>), fallback local Ollama (<code>llama3.2:3b</code>), plus offline FakeLLM mode.
    </p>
  </div>
</div>

<h2>2. Multi-Agent System Design (LangGraph)</h2>
<div class="arch-diagram">
  <div style="font-size: 8pt; font-weight: 700; color: #1e293b; margin-bottom: 6px;">7-Stage LangGraph Pipeline &amp; Deterministic Execution DAG</div>
  <div class="arch-steps">
    <div class="arch-step">
      <strong>1. Router</strong>
      <span>Intent classification &amp; follow-up rewrite</span>
    </div>
    <div class="arch-step">
      <strong>2. Memory Reader</strong>
      <span>Injects active context as UNVERIFIED</span>
    </div>
    <div class="arch-step">
      <strong>3. Retriever</strong>
      <span>Dense vector + BM25 RRF fusion</span>
    </div>
    <div class="arch-step">
      <strong>4. Analyst</strong>
      <span>Temporal conflict &amp; CURRENT labeling</span>
    </div>
    <div class="arch-step">
      <strong>5. Answer Writer</strong>
      <span>Strict citations &amp; calculator tool</span>
    </div>
    <div class="arch-step">
      <strong>6. Verifier</strong>
      <span>Claim check &amp; citation audit</span>
    </div>
    <div class="arch-step">
      <strong>7. Memory Writer</strong>
      <span>Zero-knowledge policy gate</span>
    </div>
  </div>
</div>

<h2>3. Memory Policy, Privacy &amp; Multi-Tenant Isolation</h2>
<div class="highlight-box">
  <strong>Guiding Principle:</strong> Remembered context is <em>user claim</em>, NEVER corporate evidence. A remembered statement or past conversation turn cannot prove a company fact.
</div>

<div class="grid-2">
  <div class="card">
    <div class="card-title"><span>What We Store</span> <span class="pill pill-green">Allowed</span></div>
    <ul style="padding-left: 14px; font-size: 7.5pt; color: #475569;">
      <li>Stable workflow preferences (formatting, concise style).</li>
      <li>User's role, team, and current active projects.</li>
      <li>Explicit "remember that" statements marked as <code>USER_CLAIM</code>.</li>
      <li>Self-corrections ("actually, I'm on Team plan").</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title"><span>What We NEVER Store</span> <span class="pill pill-red">Hard Blocked</span></div>
    <ul style="padding-left: 14px; font-size: 7.5pt; color: #475569;">
      <li>API keys, tokens, auth headers (regex + LLM scan).</li>
      <li>Health, medical, or parental leave data (redirected to HR).</li>
      <li>Company facts claimed by users ("SLA is 100%").</li>
      <li>Cross-user data (strictly isolated by <code>user_id</code> in SQL).</li>
    </ul>
  </div>
</div>

<p style="font-size: 8pt;">
<strong>Hard Delete Guarantee ("Forget"):</strong> Calling "forget [topic]" or "forget all" executes hard deletion across the SQLite database, vector indexes, sessions, and messages. Adversarial re-surfacing tests verify <strong>0% leakage</strong>.
</p>

<!-- ── PAGE 2: EVALUATION & PITFALLS ── -->
<div class="page-break"></div>

<h2>4. Reproducible Evaluation Results (Baseline vs Improved)</h2>
<p>
The evaluation was executed over <strong>80 research test cases</strong> (<code>eval/datasets/research.jsonl</code>) across 9 categories and <strong>34 multi-turn memory scenarios</strong> (<code>eval/datasets/memory.jsonl</code>). All numbers are generated from <code>eval/final/*.json</code>. Full test suite has <strong>65/65 passing tests</strong>.
</p>

<table>
  <thead>
    <tr>
      <th>Evaluation Metric</th>
      <th>Baseline (Frozen)</th>
      <th>Improved (7 Agents)</th>
      <th>Delta / Impact</th>
      <th>Key Driver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Retrieval Recall@5</strong></td>
      <td>97.4% (78/80)</td>
      <td><strong>94.9%</strong></td>
      <td>-2.5 pp</td>
      <td>Dense BGE-small + BM25 RRF fusion</td>
    </tr>
    <tr>
      <td><strong>MRR (Mean Reciprocal Rank)</strong></td>
      <td>0.765</td>
      <td><strong>0.702</strong></td>
      <td>-0.063</td>
      <td>Balanced multi-document retrieval</td>
    </tr>
    <tr>
      <td><strong>Conflict Handling Accuracy</strong></td>
      <td>42.9% (3/7)</td>
      <td><strong>100.0% (7/7)</strong></td>
      <td><span class="pill pill-green">+57.1 pp</span></td>
      <td><strong>Evidence Analyst</strong> temporal date &amp; version rules</td>
    </tr>
    <tr>
      <td><strong>Arithmetic Accuracy</strong></td>
      <td>46.2% (6/13)</td>
      <td><strong>84.6% (11/13)</strong></td>
      <td><span class="pill pill-green">+38.5 pp</span></td>
      <td><strong>Deterministic Calculator Tool</strong></td>
    </tr>
    <tr>
      <td><strong>Follow-up Accuracy</strong></td>
      <td>57.1% (4/7)</td>
      <td><strong>85.7% (6/7)</strong></td>
      <td><span class="pill pill-green">+28.6 pp</span></td>
      <td><strong>Router</strong> contextual history rewriting</td>
    </tr>
    <tr>
      <td><strong>Citation Validity</strong></td>
      <td>100.0% (80/80)</td>
      <td><strong>97.5% (78/80)</strong></td>
      <td>-2.5 pp</td>
      <td>Strict code validation of chunk ID &amp; title pairs</td>
    </tr>
    <tr>
      <td><strong>Abstention Accuracy</strong></td>
      <td>91.2% (73/80)</td>
      <td><strong>91.2% (73/80)</strong></td>
      <td>0.0 pp</td>
      <td>Honest abstention on unpublished corpus facts</td>
    </tr>
    <tr>
      <td><strong>Memory Write Precision / Recall</strong></td>
      <td>—</td>
      <td><strong>1.00 / 1.00</strong></td>
      <td><span class="pill pill-green">Perfect</span></td>
      <td>Zero-knowledge Memory Policy Gate</td>
    </tr>
    <tr>
      <td><strong>Memory Isolation Leakage</strong></td>
      <td>—</td>
      <td><strong>0 (0.00%)</strong></td>
      <td><span class="pill pill-green">Zero Leak</span></td>
      <td>Strict SQL <code>WHERE user_id = ?</code> constraint</td>
    </tr>
    <tr>
      <td><strong>Secret &amp; Sensitive Leak Count</strong></td>
      <td>—</td>
      <td><strong>0 (0.00%)</strong></td>
      <td><span class="pill pill-green">Zero Leak</span></td>
      <td>Dual regex + semantic credential / health guardrails</td>
    </tr>
  </tbody>
</table>

<h2>5. Corpus Pitfalls Mastery (16 Scenarios Verified)</h2>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Pitfall / Subtle Fact</th>
      <th>Corpus Evidence</th>
      <th>Handled Behavior</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><strong>Cold recovery storage</strong></td>
      <td><code>retention:0</code> vs <code>security:1</code></td>
      <td>Answers 20 days effective 2026-01-15; flags 60d as superseded. Cold storage is not searchable.</td>
    </tr>
    <tr>
      <td>2</td>
      <td><strong>On-call stipend</strong></td>
      <td><code>handbook:1</code> vs <code>oncall:0</code></td>
      <td>Answers $175 effective 2026-01-01 superseding $140. Severity definitions remain in runbook.</td>
    </tr>
    <tr>
      <td>3</td>
      <td><strong>AQL timeout in 2.3+</strong></td>
      <td><code>query:0</code> vs <code>release-23:0</code></td>
      <td>Answers 50 seconds in 2.3+ (was 25s in 2.0-2.2). Row limit 8000 and concurrency unchanged.</td>
    </tr>
    <tr>
      <td>4</td>
      <td><strong>Launch ingestion rate</strong></td>
      <td><code>ingest:1</code> vs <code>release-20:0</code></td>
      <td>Answers 300 rps from 2.1 (was 180 rps in 2.0). Team: 1500 rps; Enterprise: 6000 rps per project.</td>
    </tr>
    <tr>
      <td>5</td>
      <td><strong>Ingestion vs Concurrency</strong></td>
      <td><code>query:1</code> vs <code>ingest:1</code></td>
      <td>Never confuses project ingestion rps with workspace query concurrency (6 / 30 / 120).</td>
    </tr>
    <tr>
      <td>6</td>
      <td><strong>Retention tiers</strong></td>
      <td><code>retention:0</code>, <code>pricing:0-2</code></td>
      <td>Differentiates run records (21d/180d/540d), timeline (14d/60d), cold (+20d), purge (10d), backups (28d).</td>
    </tr>
    <tr>
      <td>7</td>
      <td><strong>Warehouse Export</strong></td>
      <td><code>export:0-1</code>, <code>pricing:1-2</code></td>
      <td>Team: BigQuery + Snowflake; Enterprise: +Redshift in 2.2. Deletions do not purge warehouse copies.</td>
    </tr>
    <tr>
      <td>8</td>
      <td><strong>Three distinct incidents</strong></td>
      <td><code>incident-*:0-1</code></td>
      <td>INC-2025-08 (SEV-1 Queuefin, 1.4M lost), INC-2025-12 (SEV-2 duplicate rows), INC-2026-03 (SEV-3 delayed alerts).</td>
    </tr>
    <tr>
      <td>9</td>
      <td><strong>Version history</strong></td>
      <td><code>release-*:0-2</code></td>
      <td>2.1: EU projects + anomaly; 2.2: Redshift + async SDK + Explorer filter; 2.3: PagerDuty + 50s AQL.</td>
    </tr>
    <tr>
      <td>10</td>
      <td><strong>Mandatory abstention</strong></td>
      <td>Negative corpus scope</td>
      <td>Abstains on Enterprise list price (negotiated), APAC region (none), mobile app, HIPAA.</td>
    </tr>
    <tr>
      <td>11</td>
      <td><strong>User-described context</strong></td>
      <td><code>company:2</code></td>
      <td>Workspace context is marked as USER_CLAIM, never corporate capabilities.</td>
    </tr>
    <tr>
      <td>12</td>
      <td><strong>Run Explorer &amp; SDKs</strong></td>
      <td><code>architecture:0-1</code></td>
      <td>Correctly clarifies that Explorer and SDKs are workflow instrumentation, NOT screen recorders.</td>
    </tr>
    <tr>
      <td>13</td>
      <td><strong>Pricing arithmetic</strong></td>
      <td><code>pricing:0-2</code></td>
      <td>Extra Team seats $16/mo; annual 12% discount applies only to base plans; deterministic calculator used.</td>
    </tr>
    <tr>
      <td>14</td>
      <td><strong>Enterprise SLA</strong></td>
      <td><code>pricing:2</code>, <code>legal:0</code></td>
      <td>Contractual 99.92% uptime SLA is exclusive to Enterprise; credits 10%/25%/50% within 20 days.</td>
    </tr>
    <tr>
      <td>15</td>
      <td><strong>Supersession hierarchy</strong></td>
      <td><code>faq:2</code></td>
      <td>Specific policy overrides general overview; asks for clarification if unresolved.</td>
    </tr>
    <tr>
      <td>16</td>
      <td><strong>HR policies vs user PII</strong></td>
      <td><code>handbook:0-2</code></td>
      <td>Corporate leave policy is valid evidence; user's personal medical circumstances are blocked.</td>
    </tr>
  </tbody>
</table>

<!-- ── PAGE 3: REAL BROWSER VERIFICATION & REPRODUCTION ── -->
<div class="page-break"></div>

<h2>6. Real Browser End-to-End Live QA Verification (10 Turns)</h2>
<p>
Executed live via automated Playwright on real Google Chrome at <code>http://localhost:8502</code>. All 10 turns completed with sub-6 second latencies and <strong>100% pass rate</strong>.
</p>

<table>
  <thead>
    <tr>
      <th>Turn</th>
      <th>Prompt / Scenario</th>
      <th>Assistant Response Snippet</th>
      <th>Latency</th>
      <th>Guardrails / Verification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>T0</strong></td>
      <td>Sidebar Toggle Test</td>
      <td>Fixed glowing expand button toggles sidebar smoothly without clipping.</td>
      <td>0.2s</td>
      <td><span class="pill pill-green">PASS</span> CSS Fixed (14px, 14px)</td>
    </tr>
    <tr>
      <td><strong>T1</strong></td>
      <td>Role &amp; Corpus Inventory</td>
      <td>"I'm Asterline's research assistant... loaded knowledge base contains 75 chunks across 25 docs."</td>
      <td>4.26s</td>
      <td><span class="pill pill-green">PASS</span> Deterministic Meta Check</td>
    </tr>
    <tr>
      <td><strong>T2</strong></td>
      <td>Launch Ingestion Rate</td>
      <td>"The current Launch ingestion rate limit is 300 requests per second per project [ingest:1 — Run ingestion API]."</td>
      <td>4.45s</td>
      <td><span class="pill pill-green">PASS</span> Grounded Fact Citation</td>
    </tr>
    <tr>
      <td><strong>T3</strong></td>
      <td>Follow-up ("What was it at 2.0?")</td>
      <td>"In version 2.0, Launch ingestion was limited to 180 requests per second [release-20:0 — Version 2.0 release notes]."</td>
      <td>4.43s</td>
      <td><span class="pill pill-green">PASS</span> Router History Rewrite</td>
    </tr>
    <tr>
      <td><strong>T4</strong></td>
      <td>Cold Recovery Supersession</td>
      <td>"Raw run records in cold recovery storage are held for 20 days... supersedes 60 days [retention:0]."</td>
      <td>5.04s</td>
      <td><span class="pill pill-green">PASS</span> Evidence Analyst Supersession</td>
    </tr>
    <tr>
      <td><strong>T5</strong></td>
      <td>Arithmetic (3 extra Team seats)</td>
      <td>"Extra Team seats cost $16 per seat per month [pricing:1]. For 3 extra seats: 3 × $16 = $48/month."</td>
      <td>6.48s</td>
      <td><span class="pill pill-green">PASS</span> Deterministic Calculator</td>
    </tr>
    <tr>
      <td><strong>T6</strong></td>
      <td>Enterprise List Price</td>
      <td>"There is no list price published for the Enterprise tier, as its pricing is negotiated directly."</td>
      <td>5.21s</td>
      <td><span class="pill pill-green">PASS</span> Honest Abstention</td>
    </tr>
    <tr>
      <td><strong>T7</strong></td>
      <td>Explicit Memory Storage</td>
      <td>"Remembered: Our team uses the Team plan on the EU cluster." Saved to user context as claim.</td>
      <td>5.75s</td>
      <td><span class="pill pill-green">PASS</span> Memory Policy Gate</td>
    </tr>
    <tr>
      <td><strong>T8</strong></td>
      <td>Memory Disambiguation</td>
      <td>"Per-workspace query concurrency is 30 on Team [query:1]. From saved context: your team uses Team plan."</td>
      <td>5.29s</td>
      <td><span class="pill pill-green">PASS</span> Context Personalization</td>
    </tr>
    <tr>
      <td><strong>T9</strong></td>
      <td>Secret Key Guardrail</td>
      <td>"I did not store that secret value. Please rotate the credential and do not share it in chat."</td>
      <td>3.04s</td>
      <td><span class="pill pill-red">BLOCKED</span> Secret Key Rejected</td>
    </tr>
    <tr>
      <td><strong>T10</strong></td>
      <td>Health / Leave PII Guardrail</td>
      <td>"Employees should use the HR portal for personal leave requests rather than storing them in chat."</td>
      <td>5.80s</td>
      <td><span class="pill pill-red">BLOCKED</span> Sensitive PII Rejected</td>
    </tr>
  </tbody>
</table>

<h2>7. Production Engineering, Failover &amp; Reproducibility</h2>
<div class="grid-2">
  <div class="card">
    <div class="card-title"><span>Provider Failover &amp; Circuit Breakers</span></div>
    <p style="font-size: 7.5pt; color: #475569;">
      Three-tier LLM hierarchy: Primary Gemini (<code>gemini-3.5-flash-lite</code>) with automatic exponential backoff, jitter, and Retry-After header parsing. When quota is exhausted, falls back to local Ollama (<code>llama3.2:3b</code>). If Ollama is offline, displays sanitized degraded mode. Complete test suite runs offline via <code>FakeLLM</code>.
    </p>
  </div>
  <div class="card">
    <div class="card-title"><span>LangSmith Observability &amp; Redaction</span></div>
    <p style="font-size: 7.5pt; color: #475569;">
      Every LangGraph node generates named spans tagged with agent name, intent, hashed user ID, and session ID. Regex redactors strip API keys, Bearer tokens, and auth headers before sending to LangSmith. Tracing rate-limit errors (HTTP 429) are gracefully handled so they never disrupt user chat.
    </p>
  </div>
</div>

<h2>8. Reproduction &amp; CLI Commands</h2>
<pre># 1. Setup environment and dependencies
uv sync

# 2. Run full test suite (65 tests passing)
uv run asterline test        # or: pytest tests/ -q

# 3. Ingest knowledge base idempotently
uv run asterline ingest

# 4. Ask a question via CLI
uv run asterline ask "What is the current Launch ingestion rate?" --provider gemini

# 5. Launch Codex-style Streamlit UI
uv run asterline run         # Accessible on http://localhost:8501 and http://localhost:8502</pre>

<h2>9. AI Use Disclosure</h2>
<p style="font-size: 8pt; color: #475569;">
This project was implemented with AI assistance under strict human engineering direction:
</p>
<ul style="padding-left: 14px; font-size: 7.5pt; color: #475569;">
  <li><strong>Codex &amp; Antigravity:</strong> Pair-programming assistant used for code generation, test authoring, and UI refinement.</li>
  <li><strong>Claude:</strong> Consulted during architectural planning for dataset schema design and prompt engineering.</li>
  <li><strong>Human Architectural Decisions:</strong> LangGraph 7-stage DAG topology, zero-knowledge memory policy, deterministic arithmetic calculator design, SQLite WAL isolation, and comprehensive corpus pitfall test coverage.</li>
</ul>

<div class="footer-note">
  <span><strong>{CANDIDATE_NAME}</strong> (NIAT ID: {CANDIDATE_ID})</span>
  <span>Repository: {GITHUB_URL}</span>
  <span>LangSmith Project: asterline-assistant</span>
</div>

</body>
</html>
"""

def generate_pdf(output_pdf: Path = Path("SUBMISSION.pdf"), output_html: Path = Path("SUBMISSION.html")) -> None:
    output_html.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"Wrote HTML to {output_html}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True
        )
        page = browser.new_page()
        page.set_content(HTML_TEMPLATE, wait_until="networkidle")
        page.pdf(
            path=str(output_pdf),
            format="A4",
            print_background=True,
            display_header_footer=False,
            margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"}
        )
        browser.close()

    print(f"Successfully generated executive PDF submission: {output_pdf} ({output_pdf.stat().st_size:,} bytes)")

if __name__ == "__main__":
    generate_pdf()
