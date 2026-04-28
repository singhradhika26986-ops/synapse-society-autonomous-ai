def build_action_explanation(agent, state, selected_action, scores, factors, memories):
    ordered_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_memories = [
        {
            "text": memory["text"][:180],
            "sentiment": memory["sentiment"],
            "timestep": memory["timestep"],
            "metadata": memory["metadata"],
        }
        for memory in memories[:3]
    ]
    return {
        "agent_id": agent.id,
        "selected_action": selected_action,
        "state_key": list(state),
        "ranked_actions": [{"action": action, "score": round(score, 3)} for action, score in ordered_scores],
        "factors": {key: round(value, 3) if isinstance(value, float) else value for key, value in factors.items()},
        "memory_influence": top_memories,
        "summary": _summary(selected_action, ordered_scores, factors),
    }


def _summary(selected_action, ordered_scores, factors):
    runner_up = ordered_scores[1][0] if len(ordered_scores) > 1 else "none"
    return (
        f"Selected {selected_action} because its weighted score was highest; "
        f"runner-up was {runner_up}. Main pressures: hunger={factors.get('hunger_pressure')}, "
        f"thirst={factors.get('thirst_pressure')}, energy_need={factors.get('energy_pressure')}, "
        f"trust={factors.get('target_trust')}, zone_risk={factors.get('zone_risk')}."
    )
