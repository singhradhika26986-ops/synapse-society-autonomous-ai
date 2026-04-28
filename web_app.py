import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from communication import Communicator, ConversationHistory
from config import DEFAULT_AGENT_COUNT, FPS
from environment import GridWorld
from logger import SimulationLogger
from metrics import SocietyMetrics


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synapse Society Live App</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Consolas, "Cascadia Mono", monospace;
      background: #14181f;
      color: #eef2f7;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(420px, 650px) minmax(360px, 1fr);
      gap: 0;
      background: #14181f;
    }
    main {
      padding: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    aside {
      background: #202631;
      border-left: 1px solid #3b4452;
      padding: 22px;
      overflow: auto;
      max-height: 100vh;
      box-sizing: border-box;
    }
    canvas {
      width: min(88vw, 640px);
      max-width: 100%;
      aspect-ratio: 1 / 1;
      background: #1a202a;
      border: 1px solid #4a5566;
    }
    h1 {
      margin: 0 0 14px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    h2 {
      margin: 24px 0 10px;
      font-size: 16px;
      color: #d7deea;
      letter-spacing: 0;
    }
    .agent {
      border-bottom: 1px solid #3b4452;
      padding: 10px 0;
    }
    .name {
      font-weight: 700;
    }
    .meta, .message {
      color: #b9c2d2;
      font-size: 13px;
      line-height: 1.45;
      margin-top: 4px;
    }
    .bar {
      height: 6px;
      background: #11151c;
      margin-top: 6px;
      position: relative;
    }
    .fill {
      height: 100%;
      width: 0%;
      background: #5ba8ff;
    }
    .hunger .fill {
      background: #4bbe78;
    }
    .thirst .fill {
      background: #48aae6;
    }
    .trust {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 7px;
    }
    .pill {
      border: 1px solid #4b5565;
      color: #d7deea;
      padding: 2px 6px;
      font-size: 12px;
      background: #161b24;
    }
    #metrics {
      display: grid;
      gap: 12px;
    }
    .metric-canvas {
      width: 100%;
      height: 120px;
      background: #151a22;
      border: 1px solid #3b4452;
    }
    .chat {
      font-size: 13px;
      line-height: 1.45;
      color: #cbd4e4;
      padding: 8px 0;
      border-bottom: 1px solid #343d4a;
    }
    @media (max-width: 860px) {
      body {
        grid-template-columns: 1fr;
      }
      aside {
        max-height: none;
        border-left: none;
        border-top: 1px solid #3b4452;
      }
    }
  </style>
</head>
<body>
  <main>
    <canvas id="world" width="640" height="640"></canvas>
  </main>
  <aside>
    <h1>Synapse Society Live</h1>
    <div id="summary" class="meta">Loading simulation...</div>
    <h2>Agents</h2>
    <div id="agents"></div>
    <h2>Learning Metrics</h2>
    <div id="metrics">
      <canvas id="rewardChart" class="metric-canvas" width="420" height="120"></canvas>
      <canvas id="trustChart" class="metric-canvas" width="420" height="120"></canvas>
      <div id="metricText" class="meta"></div>
    </div>
    <h2>Live Communication</h2>
    <div id="chat"></div>
  </aside>
  <script>
    const canvas = document.getElementById("world");
    const ctx = canvas.getContext("2d");
    const colors = ["#5ba8ff", "#ffb142", "#ec6182", "#92e073", "#bd89ff"];
    const zoneColors = {safe:"#1f3630", neutral:"#1a202a", risky:"#3a232b", high_resource:"#36321e"};
    const resourceColors = {food:"#4bbe78", water:"#48aae6", cache:"#e8c65c"};

    function draw(state) {
      const size = state.size;
      const cell = canvas.width / size;
      ctx.fillStyle = "#1a202a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (const zone of state.zones) {
          ctx.fillStyle = zoneColors[zone.type] || zoneColors.neutral;
          ctx.fillRect(zone.x * cell, zone.y * cell, cell, cell);
      }
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          ctx.strokeStyle = "#3a4350";
          ctx.lineWidth = 1;
          ctx.strokeRect(x * cell, y * cell, cell, cell);
        }
      }

      for (const obstacle of state.obstacles) {
        ctx.fillStyle = "#0a0c10";
        ctx.fillRect(obstacle.x * cell + 14, obstacle.y * cell + 14, cell - 28, cell - 28);
      }

      for (const resource of state.resources) {
        ctx.fillStyle = resourceColors[resource.type] || "#4bbe78";
        ctx.beginPath();
        ctx.arc(resource.x * cell + cell / 2, resource.y * cell + cell / 2, 8, 0, Math.PI * 2);
        ctx.fill();
      }

      for (const agent of state.agents) {
        const cx = agent.position.x * cell + cell / 2;
        const cy = agent.position.y * cell + cell / 2;
        ctx.fillStyle = colors[(agent.id - 1) % colors.length];
        ctx.beginPath();
        ctx.arc(cx, cy, 20, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#0d1016";
        ctx.font = "bold 18px Consolas";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(agent.id, cx, cy);
        ctx.fillStyle = "#5ba8ff";
        ctx.fillRect(cx - 22, cy + 28, Math.max(0, agent.energy) * 0.44, 4);
        ctx.fillStyle = "#4bbe78";
        ctx.fillRect(cx - 22, cy + 34, Math.max(0, 100 - agent.hunger) * 0.44, 4);
        ctx.fillStyle = "#48aae6";
        ctx.fillRect(cx - 22, cy + 40, Math.max(0, 100 - agent.thirst) * 0.44, 4);
      }
    }

    function renderSidebar(state) {
      document.getElementById("summary").textContent =
        `Timestep ${state.timestep} | Agents ${state.agents.length} | Resources ${state.resources.length} | Obstacles ${state.obstacles.length}`;
      document.getElementById("agents").innerHTML = state.agents.map((agent) => `
        <div class="agent">
          <div class="name" style="color:${colors[(agent.id - 1) % colors.length]}">Agent ${agent.id} [${agent.personality}/${agent.specialization}]</div>
          <div class="meta">${agent.mood} pos=(${agent.position.x}, ${agent.position.y}) action=${agent.last_action} reward=${agent.total_reward}</div>
          <div class="meta">${scoreLine(agent.decision_scores)}</div>
          <div class="bar"><div class="fill" style="width:${agent.energy}%"></div></div>
          <div class="bar hunger"><div class="fill" style="width:${100 - agent.hunger}%"></div></div>
          <div class="bar thirst"><div class="fill" style="width:${100 - agent.thirst}%"></div></div>
          <div class="meta">inventory ${inventoryLine(agent.inventory)} | survived ${agent.survival_steps}</div>
          <div class="trust">${trustPills(agent.relationships)}</div>
          <div class="message">${agent.last_message || agent.goal}</div>
        </div>
      `).join("");
      drawMetrics(state.metrics || {});
      renderMetricText(state.metrics || {});
      document.getElementById("chat").innerHTML = state.conversations.slice(-10).reverse().map((entry) => `
        <div class="chat">A${entry.speaker_id} -> A${entry.listener_id}: ${entry.message}</div>
      `).join("");
    }

    function scoreLine(scores) {
      if (!scores || Object.keys(scores).length === 0) return "scores pending";
      return Object.entries(scores).map(([key, value]) => `${key}:${value}`).join(" ");
    }

    function trustPills(relationships) {
      if (!relationships || Object.keys(relationships).length === 0) {
        return '<span class="pill">trust: none</span>';
      }
      return Object.entries(relationships).map(([id, rel]) => {
        const color = rel.trust >= 0 ? "#4bbe78" : "#ec6182";
        return `<span class="pill" style="border-color:${color}">A${id}: ${rel.trust.toFixed(2)} ${rel.status}</span>`;
      }).join("");
    }

    function inventoryLine(inventory) {
      return Object.entries(inventory || {}).map(([key, value]) => `${key}:${value}`).join(" ");
    }

    function drawMetrics(metrics) {
      drawSeries(document.getElementById("rewardChart"), metrics.reward_history || {}, -8, 16, "Reward trend");
      drawSeries(document.getElementById("trustChart"), metrics.trust_history || {}, -1, 1, "Trust evolution");
    }

    function drawSeries(canvas, seriesMap, minY, maxY, label) {
      const c = canvas.getContext("2d");
      c.clearRect(0, 0, canvas.width, canvas.height);
      c.fillStyle = "#151a22";
      c.fillRect(0, 0, canvas.width, canvas.height);
      c.strokeStyle = "#303947";
      c.strokeRect(0, 0, canvas.width, canvas.height);
      c.fillStyle = "#cbd4e4";
      c.font = "12px Consolas";
      c.fillText(label, 10, 16);
      Object.entries(seriesMap).slice(0, 8).forEach(([id, points], index) => {
        if (!points || points.length < 2) return;
        c.strokeStyle = colors[index % colors.length];
        c.beginPath();
        points.slice(-80).forEach((point, i, arr) => {
          const x = 8 + (i / Math.max(1, arr.length - 1)) * (canvas.width - 16);
          const norm = (point.value - minY) / (maxY - minY);
          const y = canvas.height - 10 - Math.max(0, Math.min(1, norm)) * (canvas.height - 30);
          if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
        });
        c.stroke();
      });
    }

    function renderMetricText(metrics) {
      const interactions = Object.entries(metrics.interaction_counts || {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
        .map(([pair, count]) => `${pair}:${count}`)
        .join(" ");
      document.getElementById("metricText").textContent = interactions ? `Interaction frequency ${interactions}` : "Interaction frequency pending";
    }

    async function tick() {
      const response = await fetch("/state", {cache: "no-store"});
      const state = await response.json();
      draw(state);
      renderSidebar(state);
    }
    tick();
    setInterval(tick, 500);
  </script>
</body>
</html>
"""


class LiveSimulation:
    def __init__(self, agent_count=DEFAULT_AGENT_COUNT, seed=None, scenario="survival"):
        if seed is not None:
            random.seed(seed)
        self.environment = GridWorld(agent_count=agent_count, scenario_name=scenario)
        self.communicator = Communicator()
        self.conversations = ConversationHistory()
        self.logger = SimulationLogger()
        self.metrics = SocietyMetrics()
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _loop(self):
        delay = 1.0 / max(1, FPS)
        while self.running:
            with self.lock:
                self.step()
            time.sleep(delay)

    def step(self):
        records = []
        self.environment.timestep += 1
        self.environment.step_environment()
        for agent in list(self.environment.agents):
            perception = agent.perceive(self.environment)
            memories = agent.retrieve_memories(perception)
            decision = agent.decide(perception, memories, self.environment)
            result, reward = agent.perform(
                decision,
                self.environment,
                self.communicator,
                self.conversations,
                self.environment.timestep,
            )
            records.append(
                {
                    "agent_id": agent.id,
                    "decision": decision,
                    "target_id": decision["target_id"],
                    "result": result,
                    "reward": reward,
                }
            )
        self.metrics.update(self.environment.timestep, self.environment, records)
        self.logger.log(
            {
                "timestep": self.environment.timestep,
                "world": self.environment.snapshot(),
                "records": records,
                "conversations": self.conversations.messages[-10:],
                "metrics": self.metrics.snapshot(),
            }
        )

    def snapshot(self):
        with self.lock:
            data = self.environment.snapshot()
            data["conversations"] = list(self.conversations.messages)
            data["metrics"] = self.metrics.snapshot()
            return data


class AppHandler(BaseHTTPRequestHandler):
    simulation = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTML, "text/html; charset=utf-8")
        elif parsed.path == "/state":
            self._send(json.dumps(self.simulation.snapshot()), "application/json")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return

    def _send(self, content, content_type):
        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Synapse Society as a local browser app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENT_COUNT, choices=range(3, 6))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", default="survival", choices=["survival", "cooperation", "competition"])
    return parser.parse_args()


def main():
    args = parse_args()
    simulation = LiveSimulation(agent_count=args.agents, seed=args.seed, scenario=args.scenario)
    simulation.start()
    AppHandler.simulation = simulation
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Synapse Society live app: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
