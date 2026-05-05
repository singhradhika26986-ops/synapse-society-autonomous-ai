import argparse
import os
import random
from typing import Optional

from communication import Communicator, ConversationHistory
from config import DEFAULT_AGENT_COUNT
from environment import GridWorld
from logger import SimulationLogger, build_runtime_logger
from metrics import SocietyMetrics


class Simulation:
    def __init__(
        self,
        agent_count=DEFAULT_AGENT_COUNT,
        max_steps=25,
        seed=None,
        ui_mode="auto",
        scenario="survival",
        log_level="INFO",
    ):
        if seed is not None:
            random.seed(seed)
        self.environment = GridWorld(agent_count=agent_count, scenario_name=scenario)
        self.communicator = Communicator()
        self.conversations = ConversationHistory()
        self.logger = SimulationLogger()
        self.runtime_logger = build_runtime_logger(log_level)
        self.metrics = SocietyMetrics()
        self.max_steps = max_steps
        self.ui_mode = ui_mode
        self.ui = self._create_ui(ui_mode)
        self.runtime_logger.info(
            "Simulation initialized | scenario=%s agents=%s ui=%s max_steps=%s seed=%s transformer=%s",
            scenario,
            agent_count,
            "pygame" if self.ui else "headless",
            self.max_steps,
            seed,
            "enabled" if self.communicator.transformer_ready else "fallback",
        )

    def _create_ui(self, ui_mode):
        if ui_mode == "headless":
            return None
        try:
            from ui import PygameUI

            return PygameUI(self.environment.size)
        except Exception as exc:
            if ui_mode == "pygame":
                raise RuntimeError("Pygame UI requested but pygame is not available.") from exc
            self.runtime_logger.warning("Pygame UI unavailable, continuing in headless mode: %s", exc)
            return None

    def step(self):
        timestep_records = []
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
            timestep_records.append(
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
        self.metrics.update(self.environment.timestep, self.environment, timestep_records)
        self.logger.log(
            {
                "timestep": self.environment.timestep,
                "world": self.environment.snapshot(),
                "records": timestep_records,
                "conversations": self.conversations.messages[-10:],
                "metrics": self.metrics.snapshot(),
            }
        )
        self._log_step_summary(timestep_records)

    def _log_step_summary(self, timestep_records):
        summary = " | ".join(
            f"A{record['agent_id']} {record['decision']['action']} reward={record['reward']:.2f} pos={record['agent_state']['position']}"
            for record in timestep_records
        )
        self.runtime_logger.info("Step %s complete | %s", self.environment.timestep, summary)
        if self.conversations.messages:
            latest = self.conversations.messages[-1]
            self.runtime_logger.info(
                "Latest interaction | step=%s A%s->A%s sentiment=%.2f message=%s",
                latest["timestep"],
                latest["speaker_id"],
                latest["listener_id"],
                latest["sentiment"],
                latest["message"],
            )

    def run(self):
        running = True
        while running:
            if self.ui and not self.ui.handle_events():
                break
            self.step()
            if self.ui:
                self.ui.draw(self.environment, self.conversations)
            if self.max_steps is not None and self.environment.timestep >= self.max_steps:
                running = False
        if self.ui:
            self.ui.close()
        self.print_demo_summary()

    def print_demo_summary(self):
        print("\n=== Synapse Society Demo Summary ===")
        print(
            f"Scenario: {self.environment.scenario_name} | Steps: {self.environment.timestep} | "
            f"Agents: {len(self.environment.agents)}"
        )
        for agent in self.environment.agents:
            relationships = agent.memory.relationship_summary()
            strongest = max(
                relationships.items(),
                key=lambda item: abs(item[1]["trust"]),
                default=("none", {"trust": 0.0, "status": "neutral"}),
            )
            print(
                f"A{agent.id} {agent.personality}/{agent.specialization} "
                f"pos={agent.position} energy={agent.energy} hunger={agent.hunger} thirst={agent.thirst} "
                f"last={agent.last_action} reward={agent.total_reward:.2f} strongest_relation=A{strongest[0]} "
                f"trust={strongest[1]['trust']:+.2f} status={strongest[1]['status']}"
            )
        if self.conversations.messages:
            print("\nRecent messages:")
            for entry in self.conversations.messages[-5:]:
                print(f"- Step {entry['timestep']}: A{entry['speaker_id']} -> A{entry['listener_id']}: {entry['message']}")
        print(f"\nStructured logs: logs/simulation_log.jsonl")

    def build_summary(self):
        agents = []
        for agent in self.environment.agents:
            relationships = agent.memory.relationship_summary()
            strongest = max(
                relationships.items(),
                key=lambda item: abs(item[1]["trust"]),
                default=("none", {"trust": 0.0, "status": "neutral"}),
            )
            agents.append(
                {
                    "agent_id": agent.id,
                    "personality": agent.personality,
                    "specialization": agent.specialization,
                    "position": agent.position,
                    "energy": agent.energy,
                    "hunger": agent.hunger,
                    "thirst": agent.thirst,
                    "last_action": agent.last_action,
                    "reward": round(agent.total_reward, 2),
                    "strongest_relation": {
                        "agent_id": strongest[0],
                        "trust": strongest[1]["trust"],
                        "status": strongest[1]["status"],
                    },
                }
            )
        return {
            "scenario": self.environment.scenario_name,
            "steps": self.environment.timestep,
            "agent_count": len(self.environment.agents),
            "agents": agents,
            "recent_messages": list(self.conversations.messages[-5:]),
            "metrics": self.metrics.snapshot(),
        }

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Synapse Society with one command.")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENT_COUNT, choices=range(3, 6))
    parser.add_argument("--steps", type=int, default=25, help="Number of timesteps before exit. Use --live for an endless run.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ui", choices=["auto", "headless", "pygame"], default="auto")
    parser.add_argument("--live", action="store_true", help="Run continuously until the UI window closes or the process is stopped.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--scenario", default="survival", choices=["survival", "cooperation", "competition"])
    return parser.parse_args()


if __name__ == "__main__":
    if not os.environ.get("SYNAPSE_ALLOW_MODEL_DOWNLOAD") and not os.environ.get("SYNAPSE_TEXT_MODEL"):
        os.environ.setdefault("SYNAPSE_DISABLE_TRANSFORMER", "1")
    args = parse_args()
    max_steps: Optional[int] = None if args.live else args.steps
    Simulation(
        agent_count=args.agents,
        max_steps=max_steps,
        seed=args.seed,
        ui_mode=args.ui,
        scenario=args.scenario,
        log_level=args.log_level,
    ).run()


def run_simulation_demo(steps=30, agent_count=DEFAULT_AGENT_COUNT, seed=None, scenario="survival", log_level="WARNING"):
    """Run a bounded headless simulation and return rich step-by-step results."""
    simulation = Simulation(
        agent_count=agent_count,
        max_steps=max(1, int(steps)),
        seed=seed,
        ui_mode="headless",
        scenario=scenario,
        log_level=log_level,
    )
    step_history = []
    while simulation.environment.timestep < simulation.max_steps:
        simulation.step()
        last_conversation = simulation.conversations.messages[-1] if simulation.conversations.messages else None
        step_history.append(
            {
                "step": simulation.environment.timestep,
                "agents": [
                    {
                        "agent_id": agent.id,
                        "action": agent.last_action,
                        "decision_scores": dict(agent.last_decision_scores),
                        "mood": agent.mood,
                        "position": agent.position,
                        "energy": agent.energy,
                        "hunger": agent.hunger,
                        "thirst": agent.thirst,
                        "message": agent.last_message,
                    }
                    for agent in simulation.environment.agents
                ],
                "latest_conversation": last_conversation,
            }
        )
    return {
        "steps_requested": steps,
        "steps_completed": simulation.environment.timestep,
        "summary": simulation.build_summary(),
        "step_history": step_history,
    }
