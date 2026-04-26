import json
import os
import threading

import gradio as gr

from agent.heuristic_agent import HeuristicAgent
from agent.llm_agent import LLMAgent
from agent.random_agent import RandomAgent
from env.environment import IncidentResponseEnv

semaphore = threading.Semaphore(1)  # max 1 concurrent episode (budget guard)


def run_episode_demo(agent_choice: str, seed: int):
    if not semaphore.acquire(blocking=False):
        yield "⏳ Demo busy — another episode running. Try in 30 seconds.", "", ""
        return

    try:
        env = IncidentResponseEnv(max_steps=30)  # headroom: 5× logs + diagnose + fixes + 2× healthy ticks

        if agent_choice == "Random":
            agent = RandomAgent()
        elif agent_choice == "Heuristic":
            agent = HeuristicAgent()
        else:
            if not os.environ.get("GROQ_API_KEY"):
                yield (
                    "**GROQ_API_KEY** is not set. Add it in HF Space secrets or your shell, then try **LLM (Groq)** again.",
                    "",
                    "",
                )
                return
            agent = LLMAgent()

        if hasattr(agent, "reset"):
            agent.reset()

        obs, _ = env.reset(seed=seed)
        log = []
        total_reward = 0.0

        log.append(f"🚨 **Incident started** | Seed: {seed}")
        log.append(f"System health: {obs['system_health_score']:.2f}")
        log.append(f"Alerts: {', '.join(obs['recent_alerts']) or 'none'}")
        log.append("---")

        done = False
        step_idx = 0
        info: dict = {}
        while not done:
            try:
                action = agent.act(obs)
            except Exception as e:
                log.append(f"**Agent error:** `{e}`")
                break
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step_idx += 1

            a_str = json.dumps(action)
            log.append(f"**Step {step_idx}** | Action: `{a_str}`")
            log.append(f"↳ {obs['last_action_result']}")
            log.append(f"↳ Health: {obs['system_health_score']:.2f} | Reward: {reward:+.1f}")
            log.append("")
            yield "\n".join(log), f"{total_reward:.1f}", ""

        outcome = info.get("outcome", "unknown")
        rc = info.get("root_cause", "?")
        fm = info.get("failure_mode", "?")
        diag = "✅" if info.get("diagnosis_correct") else "❌"

        log.append("---")
        log.append(f"**Outcome: {outcome.upper()}** | Root cause was: `{rc} / {fm}`")
        log.append(f"Diagnosis: {diag} | Total Reward: {total_reward:.1f}")
        yield "\n".join(log), f"{total_reward:.1f}", f"{outcome.upper()} — Root cause: {rc}/{fm}"

    finally:
        semaphore.release()


LEADERBOARD_MD = """
| Agent | Success Rate | Diagnosis Acc. | Mean Reward |
|-------|-------------|----------------|-------------|
| 🎲 Random | ~26% | ~2% | ~-14 |
| 🔧 Heuristic | **~100%** | **~100%** | ~52 |
| 🤖 Trained LLM | (set `GROQ_API_KEY`) | varies | varies |
"""

with gr.Blocks(title="Incident Response Agent — OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚨 Incident Response Agent\n**Meta × PyTorch Hackathon | OpenEnv Environment**")
    gr.Markdown(
        "> An LLM agent must diagnose and fix a silent production failure across 5 microservices. "
        "Reasoning quality is scored separately from the fix."
    )

    with gr.Row():
        with gr.Column(scale=2):
            agent_select = gr.Radio(
                ["Random", "Heuristic", "LLM (Groq)"],
                label="Agent",
                value="Heuristic",
            )
            seed_input = gr.Slider(0, 100, value=42, step=1, label="Episode Seed")
            run_btn = gr.Button("▶ Run Episode", variant="primary")

        with gr.Column(scale=1):
            reward_out = gr.Textbox(label="Total Reward", interactive=False)
            outcome_out = gr.Textbox(label="Outcome", interactive=False)

    episode_log = gr.Markdown(label="Episode Log")

    gr.Markdown("## Leaderboard")
    gr.Markdown(LEADERBOARD_MD)

    run_btn.click(
        fn=run_episode_demo,
        inputs=[agent_select, seed_input],
        outputs=[episode_log, reward_out, outcome_out],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
