# Synapse Society Autonomous AI Civilization

Production-ready Python multi-agent AI simulation with explainable decisions, scenario control, persistence, replay, and a live FastAPI web interface.

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

## Run Locally

Recommended production app:

```powershell
.\start_production.ps1
```

Open:

[http://127.0.0.1:8010](http://127.0.0.1:8010)

Alternative local browser app:

```powershell
.\run_web_app.ps1 --port 8002
```

Headless simulation:

```powershell
.\run_simulation.ps1 --headless --steps 100 --scenario survival
```

Pygame simulation:

```powershell
.\run_simulation.ps1 --scenario cooperation
```

## API

- `GET /` - live web UI.
- `GET /healthz` - health check.
- `GET /api/state` - full simulation state for frontend rendering.
- `GET /api/scenarios` - scenario definitions.
- `POST /api/restart` - restart/switch scenario.
- `POST /api/save` - persist current state.
- `GET /api/runs` - list saved runs.
- `POST /api/load/{run_id}` - load latest saved state as replay.
- `POST /api/replay/{run_id}/{index}` - load a replay frame.
- `POST /api/live` - resume live simulation after replay.

Example restart:

```powershell
Invoke-WebRequest -Method POST -ContentType "application/json" `
  -Body '{"scenario":"competition","agents":4}' `
  http://127.0.0.1:8010/api/restart
```

## Explainability

Every agent snapshot includes:

- `decision_scores`: weighted action scores.
- `explanation.ranked_actions`: sorted action ranking.
- `explanation.factors`: hunger, thirst, energy pressure, trust, memory sentiment, personality, specialization, zone risk, scenario weights, local resources, and Q-values.
- `explanation.memory_influence`: top retrieved memories used in decision context.
- `explanation.summary`: plain-language reason for the chosen action.

## Persistence And Replay

Runs are saved under:

```text
data/runs/<run_id>/
```

Each run contains:

- `snapshots.jsonl` - replayable simulation frames.
- `latest_state.json` - latest saved frame.

The production app autosaves every 25 timesteps and also supports manual save from the UI.

## Docker

Build:

```bash
docker build -t synapse-society .
```

Run:

```bash
docker run -p 8000:8000 synapse-society
```

Open:

```text
http://localhost:8000
```

## Cloud Deployment

Render deployment is configured with `render.yaml`.

One-click Render blueprint setup:

[https://render.com/deploy?repo=https://github.com/singhradhika26986-ops/synapse-society-autonomous-ai](https://render.com/deploy?repo=https://github.com/singhradhika26986-ops/synapse-society-autonomous-ai)

1. Push this repository to GitHub.
2. Create a new Render Blueprint or Web Service.
3. Select Docker environment.
4. Render will use `Dockerfile` and bind to `$PORT`.
5. Set optional environment variables:

```text
SCENARIO=survival
AGENT_COUNT=4
SYNAPSE_DISABLE_TRANSFORMER=1
```

Railway can also run the Dockerfile or Procfile:

```text
web: uvicorn production_app:app --host 0.0.0.0 --port ${PORT:-8000}
```

After deployment, the platform provides the public URL.

Important: `127.0.0.1` links only work on the machine where the app is running. For a public link that anyone can open anytime, deploy this repository on Render or Railway and use the generated app URL.

## NLP Mode

By default production disables transformer downloads for reliable cloud startup. To enable HuggingFace generation:

```bash
SYNAPSE_ALLOW_MODEL_DOWNLOAD=1
SYNAPSE_TEXT_MODEL=sshleifer/tiny-gpt2
```

The local fallback still generates context-aware, memory-informed messages without external downloads.

## Verified

Validated locally:

- Python compile check passed.
- FastAPI import check passed.
- `GET /healthz` passed.
- `GET /api/state` returned full state with explainability.
- `POST /api/save` persisted a run.
- `GET /api/runs` listed saved runs.
- `POST /api/replay/{run_id}/0` loaded replay.
- `POST /api/restart` switched scenario to competition.
