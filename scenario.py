SCENARIOS = {
    "survival": {
        "description": "Agents prioritize long-term survival, resources, and safe-zone recovery.",
        "weights": {
            "hunger": 1.25,
            "thirst": 1.35,
            "energy": 1.2,
            "resources": 1.2,
            "social": 0.85,
            "trust": 0.9,
            "risk": 1.35,
            "competition": 0.8,
        },
    },
    "cooperation": {
        "description": "Agents favor alliances, trust-building, and repeated cooperation.",
        "weights": {
            "hunger": 0.95,
            "thirst": 1.0,
            "energy": 1.0,
            "resources": 0.95,
            "social": 1.45,
            "trust": 1.65,
            "risk": 1.0,
            "competition": 0.55,
        },
    },
    "competition": {
        "description": "Agents compete for resources and rivalries become more behaviorally important.",
        "weights": {
            "hunger": 1.15,
            "thirst": 1.15,
            "energy": 0.95,
            "resources": 1.5,
            "social": 0.8,
            "trust": 0.75,
            "risk": 0.8,
            "competition": 1.65,
        },
    },
}


def get_scenario(name):
    return SCENARIOS.get(name, SCENARIOS["survival"])
