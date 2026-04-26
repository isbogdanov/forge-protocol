# FORGE: Self-Evolving Agent Memory With No Weight Updates

**FORGE** (Failure-Optimized Reflective Graduation and Evolution) is a staged, population-based protocol that evolves prompt-injected natural-language memory for hierarchical ReAct agents — with no gradient updates and no stronger teacher model.

Agents improve decision-making across episodes by having a dedicated reflection agent analyze failed trajectories and produce reusable knowledge artifacts (rules, few-shot examples, or both). A champion-broadcast mechanism propagates the best-discovered memory across all parallel instances between stages.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Repository Layout

```
forge-protocol/
├── run_experiment.py              # Main entry point — orchestrates Docker workers
├── experiment_forge_eval.yaml     # Example: FORGE protocol (best transfer, rules)
├── experiment_reflexion_eval.yaml # Example: Ablation (individual transfer, mixed)
├── container_requirements.txt     # Full Python dependency list (used by Docker)
├── Dockerfile                     # Container image definition
├── .env.template                  # Template for API keys — copy to .env
├── .gitignore
└── agent_base/                    # Agent code (mounted into Docker at runtime)
    ├── run_cyborg_coordinator.py  # Per-instance runner (learning + evaluation)
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
    ├── utils/
    │   ├── settings.py            # Provider/model defaults and parameters
    │   └── learning_metrics.py    # Metrics tracking per attempt/stage
    └── logs/                      # Runtime logs (generated, not tracked)
```

---

## Prerequisites

- **Docker** (tested with Docker 24+)
- **Python 3.10+** (only needed to run `run_experiment.py` on the host; all agent code runs inside Docker)
- An API key for at least one supported LLM provider (see [Providers](#providers))

---

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd forge-protocol
```

### 2. Configure your API key

```bash
cp .env.template .env
```

Open `.env` and add your key for the provider you intend to use:

```bash
# For OpenRouter (recommended — access to all models via one key):
OPENROUTER_API_KEY=sk-or-v1-...

# For direct Google AI Studio access:
GOOGLE_API_KEY=AIza...
```

### 3. Build the Docker image

```bash
docker build -t cyborg-agent:latest .
```

This installs all Python dependencies and patches the CybORG CAGE-2 environment data files. Takes ~3–5 minutes on first build; subsequent builds are cached.

### 4. Run an experiment

```bash
python run_experiment.py experiment_forge_eval.yaml
```

Experiment output appears in `experiments/<name>_<timestamp>/`.

---

## Experiment Configuration

> **Note:** Each provided YAML covers a single experimental condition. To run other conditions, copy the relevant config and adjust `learning_strategy`, `model`, or `transfer_strategy` as needed.

### The two protocols

**FORGE** (`experiment_forge_eval.yaml`) — `transfer_strategy: "best"`, 10 instances per run.
All 10 instances learn in parallel within each stage. At the end of each stage, the best-performing instance's memory is broadcast to all others (champion replacement). Instances that exceed the graduation threshold are frozen and excluded from further updates.

**Individual / Ablation** (`experiment_reflexion_eval.yaml`) — `transfer_strategy: "individual"`, 20 instances per run.
Each instance learns in complete isolation — no knowledge is shared between instances across stages. This serves as the no-broadcast ablation to measure the contribution of population-level transfer.

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
  max_attempts: 5                # Max learning attempts per stage per instance
  success_attempts: 5            # Successful attempts required to end a stage early
  agents_to_improve: ["action_chooser", "analyst", "planner"]
  max_reflection_rules: 100      # Max rules stored per agent
  max_reflection_examples: 50    # Max examples stored per agent (examples/mixed only)
num_evaluation_runs: 2           # Evaluation episodes run after training completes
```

To run a different memory representation, set `learning_strategy` to `"examples"` or `"mixed"`. To run a different model, change `provider` and `model` (see [Switching models](#switching-models) below).

### Switching models

Change `provider` and `model` in `agent_config`:

```yaml
# OpenRouter (any supported model):
provider: "openrouter"
model: "x-ai/grok-4-fast"
model: "meta-llama/llama-4-maverick"
model: "qwen/qwen3-235b-a22b-2507"
model: "gemini-2.5-flash-lite"

# Direct Google AI Studio:
provider: "google"
model: "gemini-2.5-flash-lite"

# Vertex AI:
provider: "vertex"
model: "gemini-2.5-flash-lite"
```

### Granular reflector/exemplifier overrides (optional)

To use a different model for the learning agents:

```yaml
reflector_provider: "openrouter"
reflector_model: "google/gemini-2.5-pro"
exemplifier_provider: "openrouter"
exemplifier_model: "google/gemini-2.5-pro"
```

---

## Providers

API keys are read from `.env`. The provider name in the YAML determines which key is used:

| `provider` value | Env var read | Notes |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | Recommended — single key, all models |
| `google` | `GOOGLE_API_KEY` | Google AI Studio direct |
| `vertex` | `VERTEX_API_KEY` | Google Vertex AI |
| `openai` | `OPENAI_API_KEY` | OpenAI direct |
| `groq` | `GROQ_API_KEY` | Groq inference |

---

## Output Structure

Each run creates a timestamped directory under `experiments/`:

```
experiments/forge_experiment_rules_20260426_120000/
├── workspaces/
│   ├── instance_0/
│   │   ├── definitions/          # Evolved memory YAML files (snapshot of final state)
│   │   └── logs/
│   │       ├── runs/learning/    # Per-attempt learning logs
│   │       └── runs/evaluating/  # Post-training evaluation logs
│   ├── instance_1/
│   └── ...
└── aggregated_logs/
    ├── instance_0/               # Copied logs for analysis
    ├── ...
    ├── summary.md                # Per-instance results table
    ├── evaluation_report.md      # Aggregated evaluation scores
    └── incremental_summary.md    # Stage-by-stage progress (FORGE mode)
```

### Learned memory

At the end of training, each instance's evolved knowledge lives in its `definitions/` directory — the same YAML files that were injected into the agent's prompt. For example:

```
instance_0/definitions/
├── planner/
│   ├── reflection_knowledge.yaml  # Generated rules/examples for the Planner
│   └── ...
├── action_chooser/
│   └── reflection_knowledge.yaml
└── analyst/
    └── reflection_knowledge.yaml
```

---

## Environment: CybORG CAGE-2

FORGE is evaluated on [CybORG CAGE-2](https://github.com/cage-challenge/cage-challenge-2) — a simulated network-defense environment (13-host enterprise network, 30-step horizon, automated B-line red attacker). The Dockerfile automatically installs and patches the necessary data files.
---


