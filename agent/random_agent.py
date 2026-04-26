import random

from env.simulator import SERVICES, FAILURE_MODES


class RandomAgent:
    def act(self, obs: dict) -> dict:
        action_type = random.choice(
            [
                "check_logs",
                "diagnose",
                "restart_service",
                "rollback_deploy",
                "scale_up",
                "no_op",
            ]
        )
        target = random.choice(SERVICES)
        if action_type == "diagnose":
            return {
                "type": action_type,
                "target": target,
                "failure_mode": random.choice(FAILURE_MODES),
            }
        elif action_type == "no_op":
            return {"type": action_type}
        return {"type": action_type, "target": target}
