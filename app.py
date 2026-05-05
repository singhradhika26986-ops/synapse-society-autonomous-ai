import os
import random

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - friendly failure for missing dependency
    raise RuntimeError("Streamlit is required to run this app. Install requirements.txt first.") from exc

from main import run_simulation_demo


os.environ.setdefault("SYNAPSE_DISABLE_TRANSFORMER", "1")

MAX_STEPS = 200


def render_step(step_data):
    st.markdown(f"### Step {step_data['step']}")
    for agent in step_data["agents"]:
        with st.expander(
            f"Agent {agent['agent_id']} | action={agent['action']} | mood={agent['mood']} | "
            f"energy={agent['energy']} hunger={agent['hunger']} thirst={agent['thirst']}",
            expanded=False,
        ):
            st.write(
                {
                    "position": agent["position"],
                    "decision_scores": agent["decision_scores"],
                    "message": agent["message"] or "No message",
                }
            )
    if step_data.get("latest_conversation"):
        chat = step_data["latest_conversation"]
        st.info(
            f"Conversation at step {chat['timestep']}: "
            f"A{chat['speaker_id']} -> A{chat['listener_id']} | {chat['message']}"
        )


def render_summary(summary):
    st.subheader("Final Result Summary")
    st.write(
        {
            "scenario": summary["scenario"],
            "steps_completed": summary["steps"],
            "agent_count": summary["agent_count"],
        }
    )
    for agent in summary["agents"]:
        with st.expander(
            f"Agent {agent['agent_id']} | {agent['personality']}/{agent['specialization']} | "
            f"reward={agent['reward']}",
            expanded=False,
        ):
            st.write(agent)
    if summary["recent_messages"]:
        st.markdown("#### Recent Conversations")
        for message in summary["recent_messages"]:
            st.write(
                f"Step {message['timestep']}: A{message['speaker_id']} -> "
                f"A{message['listener_id']}: {message['message']}"
            )


def main():
    st.set_page_config(page_title="Synapse Society: Autonomous AI Simulation", layout="wide")
    st.title("Synapse Society: Autonomous AI Simulation")
    st.caption("Single-URL autonomous multi-agent AI demo built with Streamlit.")

    if "results" not in st.session_state:
        st.session_state.results = None

    steps = st.number_input("Number of steps", min_value=1, max_value=MAX_STEPS, value=30, step=1)
    run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_clicked:
        seed = random.randint(1, 10_000_000)
        with st.spinner("Running simulation..."):
            st.session_state.results = run_simulation_demo(
                steps=steps,
                agent_count=4,
                seed=seed,
                scenario="survival",
                log_level="ERROR",
            )

    results = st.session_state.results
    if results:
        st.success(
            f"Simulation completed successfully. "
            f"Steps run: {results['steps_completed']} / {results['steps_requested']}"
        )
        st.subheader("Step-by-Step Output")
        for step_data in results["step_history"]:
            render_step(step_data)
        render_summary(results["summary"])


if __name__ == "__main__":
    main()
