# FORGE — Reproducibility Artifact

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Artifact for the ACM CAIS 2026 paper:
> **FORGE: Self-Evolving Agent Memory With No Weight Updates via Population Broadcast**

This repository contains the full agent implementation, experiment runner, and YAML configuration files needed to run FORGE — a staged, population-based protocol that evolves prompt-injected natural-language memory for hierarchical ReAct agents, with no gradient updates and no stronger teacher model.

## Archived Artifact

Archived artifact: https://doi.org/10.5281/zenodo.19907612
Development repository: https://github.com/isbogdanov/forge-protocol
---

## Overview

The system trains and evaluates a hierarchical ReAct agent defending a network in the **CybORG CAGE-2** environment. Agents improve decision-making across episodes by having a dedicated learning agent analyze failed trajectories and produce reusable knowledge artifacts. A champion-broadcast mechanism propagates the best-discovered memory across all parallel instances between stages.

Two protocols are provided:

| File | Protocol | Transfer | Memory strategy | Instances |
|---|---|---|---|---|
| `experiment_forge_eval.yaml` | **FORGE** | `best` — champion broadcast | `rules` | 10 |
| `experiment_reflexion_eval.yaml` | **Reflexion** (individual) | `individual` — isolated | `mixed` | 20 |

Each config covers **one experimental condition**. To run other memory representations (`rules`, `examples`, `mixed`) or other models, copy the relevant config and adjust `learning_strategy` and `model` accordingly.

---

## Repository Structure

```
forge-protocol/
├── run_experiment.py              # Experiment runner — orchestrates Docker workers
├── experiment_forge_eval.yaml     # FORGE protocol config (best transfer, rules)
├── experiment_reflexion_eval.yaml # Individual/ablation config (no broadcast, mixed)
├── container_requirements.txt     # Python dependencies (used by Docker)
├── Dockerfile                     # Container image — installs CybORG + dependencies
├── .env.template                  # API key template — copy to .env
├── .gitignore
├── LICENSE
└── agent_base/                    # Agent code (mounted into Docker at runtime)
    ├── run_cyborg_coordinator.py  # Per-instance entry point (learning + evaluation)
    ├── agents/
    │   ├── planner.py             # Top-level Planner ReAct agent
    │   ├── analyst.py             # Analyst sub-agent (host state interpretation)
    │   ├── action_chooser.py      # ActionChooser sub-agent (action ranking)
    │   ├── reflector.py           # Reflector learning agent (rules generation)
    │   ├── exemplifier.py         # Exemplifier learning agent (examples generation)
    │   └── prompts/definitions/   # Per-agent YAML definitions and memory files
    │       ├── planner/
    │       ├── analyst/
    │       ├── action_chooser/
    │       ├── reflector/
    │       └── exemplifier/
    ├── coordinators/
    │   ├── cyborg_agent_coordinator.py  # Agent lifecycle, tool dispatch
    │   ├── learning_coordinator.py      # Abort-reflect-restart inner loop
    │   └── online_learning_manager.py   # Online (async) learning mode
    ├── llm-connector/             # LLM provider abstraction (pre-initialized workspace)
    │   └── conf/                  # llm.yaml, logs.yaml, security.yaml
    └── utils/
        ├── settings.py            # Provider/model defaults and parameters
        └── learning_metrics.py    # Metrics tracking per attempt/stage
```

---

## Prerequisites

- **Docker** — the agent and CybORG run entirely inside the container.
- **Python 3.10+** — only for `run_experiment.py` (the outer orchestrator); no packages beyond the standard library and `pyyaml`.
- **API keys** — at least one LLM provider key in `.env`.

### `.env` setup

```bash
cp .env.template .env
```

Open `.env` and fill in your key(s):

```bash
OPENROUTER_API_KEY=sk-or-v1-...   # recommended — single key, access to all models
GOOGLE_API_KEY=AIza...             # for direct Google AI Studio access
```

---

## Quick Start

### 1. Build the Docker image

```bash
docker build -t cyborg-agent:latest .
```

Installs all Python dependencies and patches the CybORG CAGE-2 data files. Takes ~3–5 minutes on first build; subsequent builds are cached.

### 2. Run an experiment

```bash
python run_experiment.py experiment_forge_eval.yaml
```

Output is written to `experiments/<name>_<timestamp>/aggregated_logs/`:
- `evaluation_report.md` — per-instance reward table
- `summary.md` — aggregate statistics
- `incremental_summary.md` — stage-by-stage progress (FORGE mode)

---

## Experiment Configuration

### The two protocols

**FORGE** (`experiment_forge_eval.yaml`) — `transfer_strategy: "best"`, 10 instances per run.
All 10 instances learn in parallel within each stage. At the end of each stage, the best-performing instance's memory is broadcast to all others (champion replacement). Instances that exceed the graduation threshold are frozen and excluded from further updates.

**Reflexion (individual)** (`experiment_reflexion_eval.yaml`) — `transfer_strategy: "individual"`, 20 instances per run.
Each instance learns in complete isolation — no knowledge is shared between instances across stages. This is the no-broadcast ablation.

### Key fields

```yaml
num_instances: 10          # Number of parallel agent instances per run
max_parallel_workers: 10   # How many Docker containers run concurrently

incremental:
  enabled: true
  stages: 6                      # Number of outer-loop stages
  transfer_strategy: "best"      # "best" = champion broadcast; "individual" = isolated
  graduation_threshold: -15      # Episode return above which an instance is frozen
                                 # (omit to disable graduation)

agent_config:
  steps: 30                      # Episode horizon (CAGE-2 canonical = 30)
  provider: "openrouter"         # LLM provider (see Providers section)
  model: "google/gemini-2.5-flash-lite"
  learning_strategy: "rules"     # Memory representation: "rules" | "examples" | "mixed"
  continual_learning: true       # Enable the inner learning loop
  reward_threshold: -1.1         # Per-step reward below which reflection is triggered
  max_attempts: 3                # Max learning attempts per stage per instance
  success_attempts: 3            # Successful attempts required to end a stage early
  agents_to_improve: ["action_chooser", "analyst", "planner"]
  max_reflection_rules: 100      # Max rules stored per agent
  max_reflection_examples: 50    # Max examples stored per agent (examples/mixed only)

num_evaluation_runs: 2           # Evaluation episodes run after training completes
```

### Switching models

Change `provider` and `model` in `agent_config`:

```yaml
# Via OpenRouter (single key, all models):
provider: "openrouter"
model: "google/gemini-2.5-flash-lite"
model: "x-ai/grok-4-fast"
model: "meta-llama/llama-4-maverick"
model: "qwen/qwen3-235b-a22b-2507"

# Direct Google AI Studio:
provider: "google"
model: "gemini-2.5-flash-lite"

# Vertex AI:
provider: "vertex"
model: "gemini-2.5-flash-lite"
```

### Granular reflector/exemplifier overrides (optional)

To use a different (e.g. stronger) model for the learning agents only:

```yaml
reflector_provider: "openrouter"
reflector_model: "google/gemini-2.5-pro"
exemplifier_provider: "openrouter"
exemplifier_model: "google/gemini-2.5-pro"
```

---

## Providers

API keys are read from `.env`. The `provider` value in the YAML determines which key is used:

| `provider` | Env var | Notes |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | Recommended — single key, all models |
| `google` | `GOOGLE_API_KEY` | Google AI Studio direct |
| `vertex` | `VERTEX_API_KEY` | Google Vertex AI |
| `openai` | `OPENAI_API_KEY` | OpenAI direct |
| `groq` | `GROQ_API_KEY` | Groq inference |

> Full FORGE reproduction requires paid LLM API calls and may be expensive. 

---

## Architecture

The key design principle is **YAML-driven memory**: the agent's behavior is shaped entirely by declarative definition files — no code changes are required to switch memory strategies, models, or agent configurations.

Each acting agent (Planner, Analyst, ActionChooser) is defined by:

| File | Purpose |
|---|---|
| `core.yaml` | Agent type, tool flags, system message |
| `initial_prompt.yaml` | Per-step prompt template |
| `persistent_knowledge.yaml` | Static domain knowledge (action glossary, fixed heuristics) |
| `reflection_knowledge.yaml` | **Dynamically evolved** — rules or examples written by the learning agents |
| `reflection_examples.yaml` | Learned few-shot examples (examples/mixed strategy only) |

The learning agents (Reflector, Exemplifier) analyze failed trajectories and write new entries into `reflection_knowledge.yaml` / `reflection_examples.yaml` of the acting agents. These files are re-injected into the system prompt at the start of every new attempt, so each restart begins with the accumulated knowledge from prior failures.

---

## Output Structure

```
<experiment_name>_<timestamp>/
├── experiment_config.yaml             # Copy of the config used for this run
├── incremental_summary.md             # Stage-by-stage champion/graduation overview
├── stage_1/
│   ├── workspaces/
│   │   ├── instance_1/
│   │   │   ├── definitions/           # Evolved memory snapshot at end of stage
│   │   │   │   ├── planner/
│   │   │   │   ├── analyst/
│   │   │   │   ├── action_chooser/
│   │   │   │   └── ...
│   │   │   ├── definitions_initial/   # Memory snapshot before stage learning began
│   │   │   │   └── ...
│   │   │   ├── docker.log             # Raw Docker output for this instance
│   │   │   └── logs/                  # (runtime logs, not tracked by git)
│   │   └── instance_2/ ...
│   └── aggregated_logs/
│       ├── instance_1/
│       │   ├── runs/learning/
│       │   │   └── learning_session_<timestamp>/
│       │   │       ├── attempt_1_<time>/
│       │   │       ├── attempt_2_<time>/
│       │   │       ├── <timestamp>_console_mirror.log
│       │   │       ├── learning_metrics.json
│       │   │       ├── results.json
│       │   │       └── trajectories/
│       │   └── connector/             # LLM token usage logs
│       ├── instance_2/ ...
│       └── summary.md                 # Per-instance results table for this stage
├── stage_2/ ...
│   └── ...
└── final_evaluation/
    ├── workspaces/
    └── aggregated_logs/
        ├── evaluation_report.md       # Per-instance evaluation reward table
        ├── summary.md
        └── instance_1/ ...
```

---

> **Data availability.** The complete episode logs collected for the paper (raw console logs, token usage, per-step reward traces, and evolved memory artifacts across all experiments and evaluated episodes) are not included in this repository due to size. They may be available upon request from the authors.

---

## Environment: CybORG CAGE-2

FORGE is evaluated on [CybORG CAGE-2](https://github.com/cage-challenge/cage-challenge-2) — a simulated network-defense environment (13-host enterprise network, 30-step horizon, automated B-line red attacker). The Dockerfile automatically installs and patches the necessary data files.

---

## License

This artifact is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE) for the full text.

The CybORG CAGE-2 environment is subject to its own license; see the [CybORG repository](https://github.com/cage-challenge/cage-challenge-2) for details.
