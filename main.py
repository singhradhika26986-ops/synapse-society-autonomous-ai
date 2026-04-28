import argparse
import random

from communication import Communicator, ConversationHistory
from config import DEFAULT_AGENT_COUNT
from environment import GridWorld
from logger import SimulationLogger
from metrics import SocietyMetrics


class Simulation:
    def __init__(self, agent_count=DEFAULT_AGENT_COUNT, max_steps=None, seed=None, headless=False, scenario="survival"):
        if seed is not None:
            random.seed(seed)
        self.environment = GridWorld(agent_count=agent_count, scenario_name=scenario)
        self.communicator = Communicator()
        self.conversations = ConversationHistory()
        self.logger = SimulationLogger()
        self.metrics = SocietyMetrics()
        self.max_steps = max_steps
        self.headless = headless
        if headless:
            self.ui = None
        else:
            from ui import PygameUI

            self.ui = PygameUI(self.environment.size)

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

    def run(self):
        running = True
        while running:
            if self.ui:
                running = self.ui.handle_events()
            self.step()
            if self.ui:
                self.ui.draw(self.environment, self.conversations)
            if self.max_steps and self.environment.timestep >= self.max_steps:
                running = False
        if self.ui:
            self.ui.close()

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
    parser = argparse.ArgumentParser(description="Autonomous multi-agent AI grid-world simulation.")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENT_COUNT, choices=range(3, 6))
    parser.add_argument("--steps", type=int, default=None, help="Optional number of timesteps before exit.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Run without pygame visualization.")
    parser.add_argument("--scenario", default="survival", choices=["survival", "cooperation", "competition"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Simulation(agent_count=args.agents, max_steps=args.steps, seed=args.seed, headless=args.headless, scenario=args.scenario).run()
