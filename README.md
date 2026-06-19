# Multimodel Damage Claims Orchestrator

> Two-step de-biased VLM pipeline for automated insurance claim adjudication.  
> Blind perception → evidence-based adjudication. No prompt injection. No anchoring bias.

---

## Architecture

```mermaid
graph TB
    subgraph Input["Data Layer"]
        C[claims.csv] 
        U[user_history.csv]
        E[evidence_requirements.csv]
        IMG[(images/)]
    end

    subgraph Phase1["Phase 1: Context Building"]
        DL[data_loader.py]
        CTX[Claim Context]
    end

    subgraph Phase2["Phase 2: Blind Perception"]
        VE[vision_engine.py]
        BF[Blind Facts JSON]
    end

    subgraph Phase3["Phase 3: Adjudication"]
        JE[judge_engine.py]
        VD[14-Column Verdict]
    end

    subgraph Phase4["Orchestration"]
        M[main.py]
        CA[(.cache.json)]
        OUT[output.csv]
    end

    subgraph Eval["Evaluation"]
        EV[evaluate.py]
        R[evaluation_report.md]
    end

    C --> DL
    U --> DL
    E --> DL
    DL --> CTX
    CTX --> M
    IMG --> VE
    VE --> BF
    BF --> JE
    JE --> VD
    M --> CA
    M --> OUT
    OUT --> EV
    EV --> R
```

## Two-Step Flow

```mermaid
sequenceDiagram
    participant M as main.py
    participant DL as data_loader
    participant VE as vision_engine
    participant JE as judge_engine
    participant CA as .cache.json
    participant OUT as output.csv

    M->>DL: load claims + history + rules
    DL-->>M: context dicts
    Note over M,VE: Step 1: Blind Perception
    M->>VE: run_blind_perception(images, category)
    Note over VE: No user text exposed
    VE-->>M: blind_facts JSON
    Note over M,JE: Step 2: Adjudication
    M->>JE: run_adjudication(context, facts)
    JE-->>M: 14-column verdict
    M->>CA: cache result
    M->>OUT: write final CSV
```

## Pipeline Detail

```mermaid
flowchart LR
    subgraph Vision["Blind Perception"]
        direction TB
        V1["Image ID: img_1"] --> V2["Image ID: img_2"]
        V2 --> VN["..."]
        VN --> VOUT["{ image_quality, visible_object,<br/>visible_parts, visible_issues,<br/>estimated_severity }"]
    end

    subgraph Judge["Adjudication"]
        direction TB
        J1["blind_facts"] --> J2["user_claim"]
        J2 --> J3["history_summary"]
        J3 --> J4["evidence_rules"]
        J4 --> JOUT["{ evidence_standard_met,<br/>risk_flags, issue_type,<br/>object_part, claim_status,<br/>severity, ... }"]
    end

    Vision --> Judge
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | **Python 3.12+** | Type-safe, VLM SDK first-class support |
| CSV & Data | **pandas 2.x** | Tabular I/O, merging |
| VLM Client | **openai 1.x** | Universal OpenAI-compatible interface |
| Vision Model | **Gemini 1.5 Flash** | Free tier, multimodal, JSON mode |
| Accuracy | **scikit-learn 1.x** | Exact-match metrics |
| Threading | **concurrent.futures** | stdlib thread pool |
| Progress | **tqdm 4.x** | CLI progress bars |

**Why this stack?**  
The `openai` library works with any OpenAI-compatible endpoint (Gemini, Ollama, local proxies).  
No heavy frameworks — CSV over SQLite for 44 claims, ThreadPool over asyncio for simplicity.

---

## Project Structure

```
├── data_loader.py           Phase 1: CSV ingestion & context assembly
├── vision_engine.py         Phase 2: Blind VLM perception (no user text)
├── judge_engine.py          Phase 3: Facts vs claim adjudication
├── main.py                  Phase 4: Orchestration, cache, threading, retry
├── evaluation/
│   ├── evaluate.py          Phase 5: Accuracy metrics & report generation
│   └── evaluation_report.md
├── claims/
│   ├── claims.csv           44 claims (user_id, image_paths, user_claim, claim_object)
│   ├── user_history.csv     47 users (flags, summaries, stats)
│   ├── evidence_requirements.csv  11 evidence rules
│   ├── sample_claims.csv    20 ground-truth rows
│   └── output.csv           Header-only result template
└── .cache.json              Per-claim result cache (auto-generated)
```

---

## Threat Mitigation

| Threat | Mitigation |
|--------|-----------|
| Prompt injection | Vision step never sees user text |
| Anchoring bias | Blind perception before adjudication |
| Auto-approve instructions | Detected and flagged as `text_instruction_present` |
| Image quality issues | `image_quality` field in blind facts |
| History bias | History provided only to adjudicator, not vision |

---

## 14-Column Output

| Column | Allowed Values |
|--------|---------------|
| `evidence_standard_met` | `true`, `false` |
| `risk_flags` | `none`, `claim_mismatch`, `user_history_risk`, `text_instruction_present`, ... |
| `issue_type` | `dent`, `scratch`, `crack`, `broken_part`, `stain`, `crushed_packaging`, `torn_packaging`, `water_damage`, `none`, `unknown` |
| `object_part` | `rear_bumper`, `front_bumper`, `windshield`, `door`, `screen`, `hinge`, `keyboard`, `trackpad`, `package_corner`, `seal`, ... |
| `claim_status` | `supported`, `contradicted`, `not_enough_information` |
| `severity` | `none`, `low`, `medium`, `high`, `unknown` |

---

## Quick Start

```bash
# 1. Install
pip install pandas openai scikit-learn tqdm

# 2. Set environment
export OPENAI_API_KEY="your-gemini-api-key"
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-1.5-flash"

# 3. Run pipeline
python main.py

# 4. Evaluate
python evaluation/evaluate.py
```

**Cost:** $0.00 — Gemini 1.5 Flash free tier (1,500 req/day).

---

## Data Flow

```
claims.csv ──┐
user_history ─┤──▶ data_loader ──▶ context dict ──▶ main.py
evidence_reqs─┘                                       │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                      vision_engine     judge_engine
                                      (blind facts)     (verdict)
                                              │               │
                                              └───────┬───────┘
                                                      ▼
                                              merge → output.csv
                                                      │
                                                      ▼
                                              evaluation/report.md
```

---

## Alternatives Considered

| Approach | Why Not Chosen |
|----------|---------------|
| Single VLM call (image + text) | Hallucinates damage matching text. No injection defense. |
| OpenCV rule-based | Can't generalize across car/laptop/package. No semantic understanding. |
| RAG pipeline | Adds complexity without solving anchoring bias. |
| PostgreSQL | 44 claims don't need a database. CSV is sufficient. |
| async/await | ThreadPool is simpler for IO-bound tasks. |

---

<p align="center">
  <sub>Built with OpenCode · DeepSeek V4 Flash Free</sub>
</p>
