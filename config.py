GRID_SIZE = 10
CELL_SIZE = 64
SIDEBAR_WIDTH = 360
FPS = 4

MIN_AGENTS = 3
MAX_AGENTS = 5
DEFAULT_AGENT_COUNT = 4

PERCEPTION_RANGE = 2
INTERACTION_ENERGY_THRESHOLD = 30

START_ENERGY_RANGE = (55, 95)
START_HUNGER_RANGE = (5, 45)

REST_ENERGY_GAIN = 16
MOVE_ENERGY_COST = 5
INTERACT_ENERGY_COST = 8
HUNGER_INCREASE_PER_STEP = 4
EAT_HUNGER_REDUCTION = 35
THIRST_INCREASE_PER_STEP = 5
DRINK_THIRST_REDUCTION = 40

MEMORY_DIMENSION = 128
MEMORY_TOP_K = 5

LOG_PATH = "logs/simulation_log.jsonl"

PERSONALITIES = {
    "cooperative": {
        "social": 1.35,
        "resource": 0.95,
        "risk": 0.9,
        "trust_bias": 1.4,
        "curiosity": 0.9,
        "tone": "warm, practical, alliance-seeking",
    },
    "aggressive": {
        "social": 0.85,
        "resource": 1.2,
        "risk": 1.35,
        "trust_bias": 1.0,
        "curiosity": 1.15,
        "tone": "direct, competitive, guarded",
    },
    "selfish": {
        "social": 0.65,
        "resource": 1.45,
        "risk": 0.95,
        "trust_bias": 0.8,
        "curiosity": 0.75,
        "tone": "resource-focused, transactional, cautious",
    },
    "social": {
        "social": 1.55,
        "resource": 0.85,
        "risk": 1.0,
        "trust_bias": 1.25,
        "curiosity": 1.2,
        "tone": "expressive, curious, connection-seeking",
    },
    "analytical": {
        "social": 1.0,
        "resource": 1.1,
        "risk": 0.75,
        "trust_bias": 1.1,
        "curiosity": 1.35,
        "tone": "measured, evidence-led, strategic",
    },
}

SPECIALIZATIONS = {
    "scout": {
        "speed": 2,
        "memory_bonus": 0,
        "energy_efficiency": 0.95,
        "influence": 1.0,
        "resource_skill": 0.9,
        "description": "faster movement and wider exploration",
    },
    "archivist": {
        "speed": 1,
        "memory_bonus": 3,
        "energy_efficiency": 1.0,
        "influence": 1.0,
        "resource_skill": 1.0,
        "description": "better recall and trust continuity",
    },
    "forager": {
        "speed": 1,
        "memory_bonus": 1,
        "energy_efficiency": 0.9,
        "influence": 0.9,
        "resource_skill": 1.35,
        "description": "better resource harvesting and survival",
    },
    "mediator": {
        "speed": 1,
        "memory_bonus": 2,
        "energy_efficiency": 1.0,
        "influence": 1.45,
        "resource_skill": 0.9,
        "description": "stronger social influence and alliance formation",
    },
    "survivor": {
        "speed": 1,
        "memory_bonus": 1,
        "energy_efficiency": 0.72,
        "influence": 0.85,
        "resource_skill": 1.1,
        "description": "lower energy costs and risk tolerance",
    },
}

RESOURCE_TYPES = {
    "food": {"count": 9, "color": (75, 190, 120), "hunger": -35, "energy": 4, "value": 1.0},
    "water": {"count": 7, "color": (72, 170, 230), "thirst": -40, "energy": 2, "value": 0.9},
    "cache": {"count": 4, "color": (232, 198, 92), "hunger": -12, "thirst": -10, "energy": 10, "value": 1.4},
}

ZONE_TYPES = {
    "safe": {"energy_delta": 1, "risk": 0.0, "resource_bonus": 0.75},
    "neutral": {"energy_delta": 0, "risk": 0.0, "resource_bonus": 1.0},
    "risky": {"energy_delta": -3, "risk": 0.16, "resource_bonus": 1.2},
    "high_resource": {"energy_delta": -1, "risk": 0.06, "resource_bonus": 1.65},
}

OBSTACLE_COUNT = 10
