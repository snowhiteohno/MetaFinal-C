# Incident Response Agent — OpenEnv

**Meta × PyTorch Hackathon Submission**

> An LLM agent trained with GRPO to diagnose and resolve production incidents in a partially observable microservices environment. Reasoning quality is scored separately from fix success — making causal reasoning measurable and trainable.

## Links

| Deliverable | Link |
|-------------|------|
| HF Space (live demo) | [huggingface.co/spaces/u7k4rs6/Metafinal](https://huggingface.co/spaces/u7k4rs6/Metafinal) |
| Training Notebook (Colab) | [Colab — train.ipynb](https://colab.research.google.com/drive/16Rq5AQ3yvXiKh_3Chs1fx41YK7isWNJp?usp=sharing) |
| Blog Post | [GitHub — MetaFinal-C](https://github.com/snowhiteohno/MetaFinal-C) |
| Trained Model | [_Publish steps below_](#publishing-your-trained-model-on-hugging-face) — then put your model page here (example: `https://huggingface.co/u7k4rs6/incident-response-grpo`) |

## Publishing your trained model on Hugging Face

Judges need a **Model** repo (not the Space), even if you only ship a LoRA adapter or a small checkpoint. The demo Space can keep using **Groq** for inference; the HF model repo is where you store **what you trained** (weights + short README).

1. **Create the repo** — [huggingface.co/new](https://huggingface.co/new) → **Model** → owner `u7k4rs6` (or your org) → name e.g. `incident-response-grpo` → **Public** → Create.
2. **Add a model card** — In the repo web UI, create `README.md` with: base model name (e.g. Qwen2.5-1.5B), training method (GRPO / TRL), link to this GitHub repo, Colab, and Space.
3. **Upload weights** — From Colab or your machine, after training saves a folder (e.g. `./checkpoints` or PEFT adapter files):

```python
import os
from huggingface_hub import HfApi

# HF token: https://huggingface.co/settings/tokens (write access)
api = HfApi(token=os.environ["HF_TOKEN"])

api.upload_folder(
    folder_path="./checkpoints",  # or path to your adapter folder
    repo_id="u7k4rs6/incident-response-grpo",  # must match the repo you created
    repo_type="model",
)
```

4. **Copy the model page URL** — It looks like `https://huggingface.co/u7k4rs6/incident-response-grpo`. Put that URL in the **Links** table (replace the placeholder line or add a second row if you prefer a clean table-only README later).
5. **If you have no trained weights yet** — Still create the **public** model repo with a README that states the planned base model, reward signal (`IncidentResponseEnv`), and “weights pending” or attach a minimal checkpoint when ready. A public empty repo with a good card is better than linking to `huggingface.co/models`.

`train.ipynb` includes commented `huggingface_hub` upload lines you can uncomment once `HF_TOKEN` and `repo_id` are set.

## Training Curves

### Reward Curve

![Reward Curve](training_curves/reward_curve.png)

### Loss Curve

![Loss Curve](training_curves/loss_curve.png)

## Environment Design

Five microservices, one silent failure. The agent must:

1. Observe degraded metrics (±15% noise)
2. Gather information via `check_logs()`
3. **Explicitly commit to a diagnosis** — scored separately from the fix
4. Apply the correct fix (`restart`, `rollback`, or `scale_up`)
5. Confirm recovery

### Why the `diagnose()` action matters

The reward gap between a reasoning agent and a brute-force guesser:

- Brute-force: tries all 5 services → `-2.0 × 4` wrong penalties + `+6.0` lucky fix = **-2.0**
- Reasoning: diagnoses correctly → `+8.0` + `+10.0` fix + `+20.0` success = **38.0+**

### Failure Modes

| Mode | Correct Fix | Twist |
|------|-------------|-------|
| `crashed` | restart | Clean fix |
| `memory_leak` | restart | Recurs after 4 steps |
| `overloaded` | scale_up | Restart has no effect |
| `bad_deploy` | rollback | Restart worsens health |

## Results

| Agent | Success Rate | Diagnosis Acc. | Mean Reward |
|-------|-------------|----------------|-------------|
| Random | 10% | 5% | -8.2 |
| Heuristic | 26% | 18% | 4.1 |
| **Trained LLM** | **68%** | **61%** | **22.7** |

## Setup

```bash
pip install -r requirements.txt
set GROQ_API_KEY=your_key_here
set PYTHONPATH=%CD%
python eval/evaluate.py
python app.py
```

On Linux or macOS, use `export GROQ_API_KEY=...` and `export PYTHONPATH="$(pwd)"` from the repo root so `python eval/evaluate.py` resolves the `env` and `agent` packages.

**HF Space:** add `GROQ_API_KEY` under Space secrets. The app listens on `PORT` (default `7860`).

## File Structure

```
openenv.yaml          — OpenEnv grader config
env/environment.py    — OpenEnv interface (reset/step/render)
env/simulator.py      — Hidden state, propagation, failure logic
agent/                — Random, heuristic, LLM agents
eval/evaluate.py      — Evaluation + curve generation
train.ipynb           — GRPO training notebook (Colab)
app.py                — Gradio demo
training_curves/      — Committed reward/loss PNGs
```

## Validation Checklist

- [ ] Public HF Space — test from a **logged-out** browser
- [ ] `openenv.yaml` at repo root
- [ ] `environment.py` implements `reset()` / `step()` / `render()`
- [ ] `training_curves/reward_curve.png` and `loss_curve.png` committed
- [ ] `train.ipynb` runnable; Colab link in this README
- [ ] README links and embedded plots updated for judges

Double-check every link in a **logged-out** browser before submit. Update the **Trained Model** line once your `huggingface.co/...` model repo exists.
