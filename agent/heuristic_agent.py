from env.simulator import SERVICES


class HeuristicAgent:
    """Diagnoses highest-CPU service, then restarts. ~22-30% success rate."""

    def __init__(self):
        self._diagnosed = False
        self._checked = set()
        self._suspected = None

    def reset(self):
        self._diagnosed = False
        self._checked = set()
        self._suspected = None

    def act(self, obs: dict) -> dict:
        metrics = obs["metrics"]
        if not self._diagnosed:
            # Check logs on top 2 CPU services
            by_cpu = sorted(SERVICES, key=lambda s: metrics[s]["cpu"], reverse=True)
            for svc in by_cpu[:2]:
                if svc not in self._checked:
                    self._checked.add(svc)
                    return {"type": "check_logs", "target": svc}
            # Diagnose highest CPU as root cause, guess memory_leak
            self._suspected = by_cpu[0]
            self._diagnosed = True
            return {"type": "diagnose", "target": self._suspected, "failure_mode": "crashed"}

        # Try restart first, then rollback, then scale
        if self._suspected:
            return {"type": "restart_service", "target": self._suspected}
        return {"type": "no_op"}
