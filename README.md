# Synapse Society Autonomous AI Civilization

Professional Streamlit-based multi-agent AI simulation with explainable decisions, step-by-step output, and deployment-ready single-URL demo flow.

Repository:

[https://github.com/singhradhika26986-ops/synapse-society-autonomous-ai](https://github.com/singhradhika26986-ops/synapse-society-autonomous-ai)

## What It Does

Synapse Society simulates 3-5 autonomous agents in a 10x10 grid world. Agents compete and cooperate while managing hunger, thirst, energy, trust, risk, resources, and long-term survival.

Core capabilities:

- Autonomous agents with personality, specialization, memory, relationships, inventory, goals, and Q-learning.
- Rich environment with obstacles, zones, and multiple resources: food, water, cache.
- Scenario modes: `survival`, `cooperation`, `competition`.
- Explainability for every decision, including ranked action scores and factor breakdowns.
- Relationship evolution with trust, hostility, alliances, and rivalries.
- Persistence to JSON/JSONL under `data/runs`.
- Replay/loading of previous saved simulation frames.
- FastAPI backend with live frontend and API endpoints.
- Docker and Render deployment configuration.

## Architecture

- `agents.py` - agent state, weighted decision model, specialization effects, action execution.
- `environment.py` - grid, zones, resources, obstacles, spawning, snapshots.
- `memory.py` - FAISS/vector memory and relationship/trust tracking.
- `learning.py` - Q-learning and reward primitives.
- `scenario.py` - scenario mode definitions and reward/behavior weights.
- `explainability.py` - interpretable action explanations.
- `metrics.py` - reward, trust, interaction, survival, action, and zone metrics.
- `persistence.py` - saved runs and replay frames.
- `production_app.py` - FastAPI production web app and API.
- `web_app.py` - lightweight stdlib local web app.
- `main.py` - pygame/headless simulation runner.
- `ui.py` - pygame visualization.

## How To Run

Python compatibility:

- Streamlit app: Python `3.9+`
- On Python `3.14`, the project still runs because it falls back to pure-Python memory if optional packages are unavailable

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py --server.port=10000 --server.address=0.0.0.0
```

What the app does:

- shows a clean web UI
- runs a bounded simulation when you click `Run Simulation`
- displays step-by-step agent actions and decision scores
- shows a final result summary
- avoids infinite loops by enforcing a max step limit

## Explainability

Every agent snapshot includes:

- `decision_scores`: weighted action scores.
- `explanation.ranked_actions`: sorted action ranking.
- `explanation.factors`: hunger, thirst, energy pressure, trust, memory sentiment, personality, specialization, zone risk, scenario weights, local resources, and Q-values.
- `explanation.memory_influence`: top retrieved memories used in decision context.
- `explanation.summary`: plain-language reason for the chosen action.

## Cloud Deployment

This project is optimized for a single Streamlit web service.

Recommended Render start command:

```text
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Verified

Validated locally:

- Python compile check passed.
- `python main.py` bounded simulation path passed.
- Streamlit entrypoint code compiled.
