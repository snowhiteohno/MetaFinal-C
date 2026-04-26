import json
from typing import Optional

from env.simulator import Simulator, SERVICES, FAILURE_MODES


class IncidentResponseEnv:
    """OpenEnv-compliant environment."""

    metadata = {"render_modes": ["human", "json"]}
    env_id = "incident-response-v1"

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.sim = Simulator()
        self._reset_episode_state()

    def _reset_episode_state(self):
        self._step = 0
        self._done = False
        self._diagnosis_made = False
        self._diagnosis_correct = False
        self._prev_health = {s: 1.0 for s in SERVICES}
        self._consecutive_healthy = 0
        self._consecutive_collapsed = 0
        self._last_action_result = "Episode started. Observe the system."
        self._health_history = []

    def reset(self, seed=None, options=None):
        self._reset_episode_state()
        self.sim.reset(seed=seed)
        obs = self._get_observation()
        return obs, {}

    def step(self, action: dict):
        assert not self._done, "Episode done — call reset()"

        reward = 0.0
        info = {}
        action_type = action.get("type", "no_op")
        target = action.get("target", None)
        failure_mode = action.get("failure_mode", None)

        prev_health = self.sim.system_health

        # --- Reward logic ---
        if action_type == "diagnose" and not self._diagnosis_made:
            self._diagnosis_made = True
            s = self.sim.state
            if target == s.root_cause and failure_mode == s.failure_mode:
                reward += 8.0
                self._diagnosis_correct = True
                self._last_action_result = f"Diagnosis: {target} / {failure_mode} — CORRECT"
            else:
                reward -= 2.0
                self._last_action_result = f"Diagnosis: {target} / {failure_mode} — INCORRECT"

        elif action_type in ("restart_service", "rollback_deploy", "scale_up"):
            if target and self.sim.state.lagged_health.get(target, 0) > 0.85:
                reward -= 1.5  # fixing healthy service
            effective, msg = self.sim.apply_fix(action_type, target)
            self._last_action_result = msg
            if effective:
                reward += 10.0 if self._diagnosis_correct else 6.0
            else:
                if target != self.sim.state.root_cause:
                    reward -= 2.0

        elif action_type == "enable_circuit_breaker":
            msg = self.sim.enable_circuit_breaker(target)
            self._last_action_result = msg

        elif action_type == "check_logs":
            hints = self.sim.get_log_hints(target, self._step)
            self._last_action_result = "LOGS:\n" + "\n".join(hints)

        elif action_type == "no_op":
            reward -= 0.5
            self._last_action_result = "No action taken."

        # Tick simulator
        health_before_tick = {s: self.sim.state.true_health[s] for s in SERVICES}
        self.sim.tick()
        self._step += 1
        self._prev_health = health_before_tick

        # Health delta reward (capped)
        new_health = self.sim.system_health
        delta = new_health - prev_health
        if delta > 0:
            reward += min(delta * 2.0, 3.0)

        # Termination checks
        system_h = self.sim.system_health
        self._health_history.append(system_h)

        # Just below 0.90: after a correct overloaded fix the root is capped at 0.90 and the
        # noisy mean can sit at ~0.895–0.899 for several ticks while dependents catch up.
        if system_h >= 0.888:
            self._consecutive_healthy += 1
        else:
            self._consecutive_healthy = 0

        if system_h <= 0.10:
            self._consecutive_collapsed += 1
        else:
            self._consecutive_collapsed = 0

        success = self._consecutive_healthy >= 2
        collapsed = self._consecutive_collapsed >= 3
        timeout = self._step >= self.max_steps

        if success:
            efficiency_bonus = (self.max_steps - self._step) * 0.3
            reward += 20.0 + efficiency_bonus
            self._done = True
            info["outcome"] = "success"
        elif collapsed:
            reward -= 15.0
            self._done = True
            info["outcome"] = "collapsed"
        elif timeout:
            self._done = True
            info["outcome"] = "timeout"

        info["diagnosis_correct"] = self._diagnosis_correct
        info["root_cause"] = self.sim.state.root_cause
        info["failure_mode"] = self.sim.state.failure_mode

        obs = self._get_observation()
        return obs, reward, self._done, False, info

    def _get_observation(self) -> dict:
        return {
            "step": self._step,
            "max_steps": self.max_steps,
            "system_health_score": round(self.sim.system_health, 3),
            "metrics": self.sim.get_noisy_metrics(),
            "metric_trend": self.sim.get_trends(self._prev_health),
            "recent_alerts": self._get_alerts(),
            "last_action_result": self._last_action_result,
            "diagnosis_made": self._diagnosis_made,
            "services": SERVICES,
            "failure_modes": FAILURE_MODES,
            "valid_actions": [
                "check_logs(service)",
                "diagnose(service, failure_mode)",
                "restart_service(service)",
                "rollback_deploy(service)",
                "scale_up(service)",
                "enable_circuit_breaker(service)",
                "no_op()",
            ],
        }

    def _get_alerts(self) -> list[str]:
        alerts = []
        for svc, h in self.sim.state.true_health.items():
            if h < 0.5:
                alerts.append(f"{svc}: health critical ({h:.2f})")
            elif h < 0.75:
                alerts.append(f"{svc}: degraded ({h:.2f})")
        return alerts[:4]  # cap at 4

    def render(self, mode="human") -> str:
        obs = self._get_observation()
        return json.dumps(obs, indent=2)

    def close(self):
        pass
