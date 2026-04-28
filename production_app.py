import os
import random
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from communication import Communicator, ConversationHistory
from config import DEFAULT_AGENT_COUNT, FPS
from environment import GridWorld
from logger import SimulationLogger
from metrics import SocietyMetrics
from persistence import PersistenceManager
from scenario import SCENARIOS

if not os.environ.get("SYNAPSE_ALLOW_MODEL_DOWNLOAD") and not os.environ.get("SYNAPSE_TEXT_MODEL"):
    os.environ.setdefault("SYNAPSE_DISABLE_TRANSFORMER", "1")


class RestartRequest(BaseModel):
    scenario: str = "survival"
    agents: int = DEFAULT_AGENT_COUNT
    seed: Optional[int] = None


class ProductionSimulation:
    def __init__(self, agent_count=DEFAULT_AGENT_COUNT, scenario="survival", seed=None):
        self.lock = threading.Lock()
        self.persistence = PersistenceManager()
        self.run_id = None
        self.replay_state = None
        self.running = False
        self.thread = None
        self._create(agent_count, scenario, seed)

    def _create(self, agent_count, scenario, seed=None):
        if seed is not None:
            random.seed(seed)
        self.environment = GridWorld(agent_count=agent_count, scenario_name=scenario)
        self.communicator = Communicator()
        self.conversations = ConversationHistory(limit=80)
        self.logger = SimulationLogger()
        self.metrics = SocietyMetrics()
        self.run_id = f"{scenario}-{int(time.time())}"
        self.replay_state = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        delay = 1.0 / max(1, FPS)
        while self.running:
            with self.lock:
                if self.replay_state is None:
                    self.step()
            time.sleep(delay)

    def restart(self, agent_count=DEFAULT_AGENT_COUNT, scenario="survival", seed=None):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        with self.lock:
            self._create(agent_count, scenario, seed)
        return self.snapshot()

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
                    "perceived_agents": [
                        {"id": item["agent"].id, "distance": item["distance"]} for item in perception
                    ],
                    "decision": self._serialize_decision(decision),
                    "result": result,
                    "reward": reward,
                    "agent_state": agent.snapshot(),
                }
            )
        self.metrics.update(self.environment.timestep, self.environment, records)
        state = self.snapshot(include_lock=False)
        state["records"] = records
        self.logger.log(state)
        if self.environment.timestep % 25 == 0:
            self.persistence.save_snapshot(state, records=records, run_id=self.run_id)

    def snapshot(self, include_lock=True):
        if include_lock:
            with self.lock:
                return self._snapshot_unlocked()
        return self._snapshot_unlocked()

    def _snapshot_unlocked(self):
        if self.replay_state is not None:
            replay = dict(self.replay_state)
            replay["mode"] = "replay"
            return replay
        state = self.environment.snapshot()
        state["mode"] = "live"
        state["run_id"] = self.run_id
        state["scenarios"] = SCENARIOS
        state["conversations"] = list(self.conversations.messages)
        state["metrics"] = self.metrics.snapshot()
        state["explanations"] = {str(agent.id): agent.last_explanation for agent in self.environment.agents}
        return state

    def save(self):
        with self.lock:
            return self.persistence.save_snapshot(self._snapshot_unlocked(), run_id=self.run_id)

    def load_replay(self, run_id):
        loaded = self.persistence.load_latest(run_id)
        if not loaded:
            return None
        with self.lock:
            self.replay_state = loaded["state"]
        return self.snapshot()

    def replay_frame(self, run_id, index):
        frame = self.persistence.replay_frame(run_id, index)
        if not frame:
            return None
        with self.lock:
            self.replay_state = frame["frame"]["state"]
            self.replay_state["replay_index"] = frame["index"]
            self.replay_state["replay_count"] = frame["count"]
        return self.snapshot()

    def resume_live(self):
        with self.lock:
            self.replay_state = None
        return self.snapshot()

    def _serialize_decision(self, decision):
        clean = dict(decision)
        clean["state_key"] = list(clean["state_key"])
        clean["memories_used"] = [
            {
                "text": memory["text"],
                "sentiment": memory["sentiment"],
                "timestep": memory["timestep"],
                "distance": memory["distance"],
                "metadata": memory["metadata"],
            }
            for memory in clean["memories_used"]
        ]
        return clean


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synapse Society Production</title>
  <style>
    body{margin:0;background:#14181f;color:#eef2f7;font-family:Consolas,monospace;display:grid;grid-template-columns:minmax(420px,650px) minmax(420px,1fr);min-height:100vh}
    main{padding:20px;display:flex;align-items:flex-start;justify-content:center}.panel{background:#202631;border-left:1px solid #3b4452;padding:18px;overflow:auto;max-height:100vh}
    canvas{width:min(88vw,640px);aspect-ratio:1/1;border:1px solid #4a5566;background:#1a202a}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
    button,select,input{background:#151a22;color:#eef2f7;border:1px solid #4a5566;padding:7px 9px;font-family:inherit}button{cursor:pointer}
    h1{font-size:22px;margin:0 0 12px}h2{font-size:16px;margin:20px 0 8px}.meta,.message{color:#b9c2d2;font-size:13px;line-height:1.45;margin-top:4px}
    .agent{border-bottom:1px solid #3b4452;padding:10px 0}.name{font-weight:700}.bar{height:6px;background:#11151c;margin-top:6px}.fill{height:100%;background:#5ba8ff}.hunger .fill{background:#4bbe78}.thirst .fill{background:#48aae6}
    .pill{display:inline-block;border:1px solid #4b5565;padding:2px 6px;margin:3px 4px 0 0;font-size:12px;background:#161b24}.chat{font-size:13px;color:#cbd4e4;padding:7px 0;border-bottom:1px solid #343d4a}
    .metric{width:100%;height:120px;background:#151a22;border:1px solid #3b4452;margin-top:8px}.explain{background:#151a22;border:1px solid #3b4452;padding:8px;margin-top:8px;font-size:12px;white-space:pre-wrap}
    @media(max-width:900px){body{grid-template-columns:1fr}.panel{border-left:0;border-top:1px solid #3b4452;max-height:none}}
  </style>
</head>
<body>
<main><canvas id="world" width="640" height="640"></canvas></main>
<section class="panel">
  <h1>Synapse Society Production</h1>
  <div class="row">
    <select id="scenario"><option>survival</option><option>cooperation</option><option>competition</option></select>
    <input id="agents" type="number" min="3" max="5" value="4">
    <button onclick="restart()">Restart</button><button onclick="saveRun()">Save</button><button onclick="resumeLive()">Live</button>
  </div>
  <div id="summary" class="meta"></div>
  <h2>Agents</h2><div id="agentsList"></div>
  <h2>Learning Metrics</h2><canvas id="rewardChart" class="metric" width="440" height="120"></canvas><canvas id="trustChart" class="metric" width="440" height="120"></canvas><div id="metricText" class="meta"></div>
  <h2>Replay / Persistence</h2><div class="row"><select id="runs"></select><button onclick="loadRun()">Load Latest</button><button onclick="loadFrame(0)">First Frame</button></div><div id="saveStatus" class="meta"></div>
  <h2>Explainability</h2><div id="explain" class="explain"></div>
  <h2>Live Conversations</h2><div id="chat"></div>
</section>
<script>
const canvas=document.getElementById('world'),ctx=canvas.getContext('2d');
const colors=['#5ba8ff','#ffb142','#ec6182','#92e073','#bd89ff'],zones={safe:'#1f3630',neutral:'#1a202a',risky:'#3a232b',high_resource:'#36321e'},res={food:'#4bbe78',water:'#48aae6',cache:'#e8c65c'};
let current=null;
async function api(path,opts){const r=await fetch(path,{cache:'no-store',headers:{'Content-Type':'application/json'},...(opts||{})}); if(!r.ok) throw new Error(await r.text()); return r.json();}
async function tick(){current=await api('/api/state'); render(current);}
function draw(s){const cell=canvas.width/s.size;ctx.fillStyle='#1a202a';ctx.fillRect(0,0,canvas.width,canvas.height);(s.zones||[]).forEach(z=>{ctx.fillStyle=zones[z.type]||zones.neutral;ctx.fillRect(z.x*cell,z.y*cell,cell,cell)});for(let y=0;y<s.size;y++)for(let x=0;x<s.size;x++){ctx.strokeStyle='#3a4350';ctx.strokeRect(x*cell,y*cell,cell,cell)};(s.obstacles||[]).forEach(o=>{ctx.fillStyle='#080a0d';ctx.fillRect(o.x*cell+14,o.y*cell+14,cell-28,cell-28)});(s.resources||[]).forEach(r=>{ctx.fillStyle=res[r.type]||'#fff';ctx.beginPath();ctx.arc(r.x*cell+cell/2,r.y*cell+cell/2,8,0,Math.PI*2);ctx.fill()});(s.agents||[]).forEach(a=>{const cx=a.position.x*cell+cell/2,cy=a.position.y*cell+cell/2;ctx.fillStyle=colors[(a.id-1)%colors.length];ctx.beginPath();ctx.arc(cx,cy,20,0,Math.PI*2);ctx.fill();ctx.fillStyle='#0d1016';ctx.font='bold 18px Consolas';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(a.id,cx,cy);ctx.fillStyle='#5ba8ff';ctx.fillRect(cx-22,cy+28,Math.max(0,a.energy)*.44,4);ctx.fillStyle='#4bbe78';ctx.fillRect(cx-22,cy+34,Math.max(0,100-a.hunger)*.44,4);ctx.fillStyle='#48aae6';ctx.fillRect(cx-22,cy+40,Math.max(0,100-a.thirst)*.44,4)})}
function render(s){draw(s);document.getElementById('scenario').value=s.scenario||'survival';document.getElementById('summary').textContent=`${s.mode} | run ${s.run_id||'replay'} | step ${s.timestep} | scenario ${s.scenario} | resources ${(s.resources||[]).length}`;document.getElementById('agentsList').innerHTML=(s.agents||[]).map(a=>`<div class="agent"><div class="name" style="color:${colors[(a.id-1)%colors.length]}">A${a.id} ${a.personality}/${a.specialization}</div><div class="meta">${a.mood} action=${a.last_action} reward=${a.total_reward} survived=${a.survival_steps}</div><div class="meta">${scores(a.decision_scores)}</div><div class="bar"><div class="fill" style="width:${a.energy}%"></div></div><div class="bar hunger"><div class="fill" style="width:${100-a.hunger}%"></div></div><div class="bar thirst"><div class="fill" style="width:${100-a.thirst}%"></div></div><div class="meta">inventory ${inv(a.inventory)}</div><div>${trust(a.relationships)}</div></div>`).join('');document.getElementById('chat').innerHTML=(s.conversations||[]).slice(-12).reverse().map(c=>`<div class="chat">A${c.speaker_id} -> A${c.listener_id}: ${c.message}</div>`).join('');document.getElementById('explain').textContent=JSON.stringify(s.explanations||{},null,2);drawSeries('rewardChart',s.metrics?.reward_history||{},-8,16,'Reward trend');drawSeries('trustChart',s.metrics?.trust_history||{},-1,1,'Trust evolution');document.getElementById('metricText').textContent='Interactions '+Object.entries(s.metrics?.interaction_counts||{}).map(([k,v])=>`${k}:${v}`).join(' ');}
function scores(o){return Object.entries(o||{}).map(([k,v])=>`${k}:${v}`).join(' ')} function inv(o){return Object.entries(o||{}).map(([k,v])=>`${k}:${v}`).join(' ')} function trust(o){return Object.entries(o||{}).map(([id,r])=>`<span class="pill">A${id}: ${r.trust.toFixed(2)} ${r.status}</span>`).join('')||'<span class="pill">trust none</span>'}
function drawSeries(id,map,min,max,label){const c=document.getElementById(id).getContext('2d'),w=c.canvas.width,h=c.canvas.height;c.clearRect(0,0,w,h);c.fillStyle='#151a22';c.fillRect(0,0,w,h);c.strokeStyle='#303947';c.strokeRect(0,0,w,h);c.fillStyle='#cbd4e4';c.font='12px Consolas';c.fillText(label,10,16);Object.entries(map).slice(0,8).forEach(([k,p],idx)=>{if(!p||p.length<2)return;c.strokeStyle=colors[idx%colors.length];c.beginPath();p.slice(-80).forEach((pt,i,arr)=>{const x=8+i/Math.max(1,arr.length-1)*(w-16),y=h-10-Math.max(0,Math.min(1,(pt.value-min)/(max-min)))*(h-30);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()})}
async function restart(){await api('/api/restart',{method:'POST',body:JSON.stringify({scenario:document.getElementById('scenario').value,agents:Number(document.getElementById('agents').value)})}); await refreshRuns();}
async function saveRun(){const r=await api('/api/save',{method:'POST'});document.getElementById('saveStatus').textContent=`saved ${r.run_id} (${r.snapshot_count} frames)`;await refreshRuns();}
async function refreshRuns(){const runs=await api('/api/runs');document.getElementById('runs').innerHTML=runs.map(r=>`<option value="${r.run_id}">${r.run_id} (${r.snapshot_count})</option>`).join('')}
async function loadRun(){const id=document.getElementById('runs').value;if(id)render(await api('/api/load/'+id,{method:'POST'}));}
async function loadFrame(i){const id=document.getElementById('runs').value;if(id)render(await api(`/api/replay/${id}/${i}`,{method:'POST'}));}
async function resumeLive(){render(await api('/api/live',{method:'POST'}));}
refreshRuns();tick();setInterval(tick,750);
</script>
</body></html>"""


app = FastAPI(title="Synapse Society Autonomous AI Civilization")
simulation = ProductionSimulation(
    agent_count=int(os.environ.get("AGENT_COUNT", DEFAULT_AGENT_COUNT)),
    scenario=os.environ.get("SCENARIO", "survival"),
    seed=int(os.environ["SEED"]) if os.environ.get("SEED") else None,
)
simulation.start()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/healthz")
def healthz():
    return {"ok": True, "scenario": simulation.environment.scenario_name}


@app.get("/api/state")
def state():
    return simulation.snapshot()


@app.get("/api/scenarios")
def scenarios():
    return SCENARIOS


@app.post("/api/restart")
def restart(request: RestartRequest):
    try:
        return simulation.restart(agent_count=request.agents, scenario=request.scenario, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/save")
def save():
    return simulation.save()


@app.get("/api/runs")
def runs():
    return simulation.persistence.list_runs()


@app.post("/api/load/{run_id}")
def load(run_id: str):
    loaded = simulation.load_replay(run_id)
    if not loaded:
        raise HTTPException(status_code=404, detail="Run not found")
    return loaded


@app.post("/api/replay/{run_id}/{index}")
def replay(run_id: str, index: int):
    frame = simulation.replay_frame(run_id, index)
    if not frame:
        raise HTTPException(status_code=404, detail="Replay frame not found")
    return frame


@app.post("/api/live")
def live():
    return simulation.resume_live()
