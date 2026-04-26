from env.simulator import SERVICES


class HeuristicAgent:
    """Uses check_logs text to guess service + failure_mode, then applies the matching fix."""

    def __init__(self):
        self._diagnosed = False
        self._checked: set[str] = set()
        self._suspected = None
        self._failure_mode_guess = "crashed"
        self._log_by_service: dict[str, str] = {}
        self._last_check_target: str | None = None

    def reset(self):
        self._diagnosed = False
        self._checked = set()
        self._suspected = None
        self._failure_mode_guess = "crashed"
        self._log_by_service = {}
        self._last_check_target = None

    def _capture_previous_logs(self, obs: dict) -> None:
        lr = obs.get("last_action_result", "")
        if self._last_check_target and lr.startswith("LOGS"):
            self._log_by_service[self._last_check_target] = lr
        self._last_check_target = None

    @staticmethod
    def _score_incident_keywords(text: str) -> int:
        """Higher = more likely this service is the real root (template lines vs generic noise)."""
        if not text:
            return 0
        t = text.lower()
        score = 0
        for kw in (
            "rollback available",
            "config parse",
            "startup probe failing",
            "deploy",
            "queue depth",
            "throttling",
            "latency spike",
            "dropping connections",
            "heap usage",
            "gc pressure",
            "memory usage at",
            "exited unexpectedly",
            "oom kill",
            "segfault",
        ):
            if kw in t:
                score += 4
        return score

    @staticmethod
    def _infer_failure_mode(combined_logs: str) -> str:
        t = combined_logs.lower()
        if any(
            x in t
            for x in (
                "rollback available",
                "config parse",
                "startup probe failing",
                "deploy",
            )
        ):
            return "bad_deploy"
        if any(x in t for x in ("queue depth", "throttling", "latency spike", "dropping connections")):
            return "overloaded"
        if any(x in t for x in ("heap usage", "gc pressure", "memory usage at")):
            return "memory_leak"
        return "crashed"

    def _pick_suspect(self, by_cpu: list[str]) -> str:
        best, best_score = None, -1
        for svc in self._checked:
            sc = self._score_incident_keywords(self._log_by_service.get(svc, ""))
            if sc > best_score:
                best_score = sc
                best = svc
        if best is not None and best_score > 0:
            return best
        return by_cpu[0]

    def act(self, obs: dict) -> dict:
        self._capture_previous_logs(obs)
        metrics = obs["metrics"]

        if not self._diagnosed:
            by_cpu = sorted(SERVICES, key=lambda s: metrics[s]["cpu"], reverse=True)
            for svc in by_cpu[:2]:
                if svc not in self._checked:
                    self._checked.add(svc)
                    self._last_check_target = svc
                    return {"type": "check_logs", "target": svc}

            combined = "\n".join(self._log_by_service.values())
            self._suspected = self._pick_suspect(by_cpu)
            self._failure_mode_guess = self._infer_failure_mode(combined)
            self._diagnosed = True
            return {
                "type": "diagnose",
                "target": self._suspected,
                "failure_mode": self._failure_mode_guess,
            }

        if not self._suspected:
            return {"type": "no_op"}

        fm = self._failure_mode_guess
        if fm == "bad_deploy":
            return {"type": "rollback_deploy", "target": self._suspected}
        if fm == "overloaded":
            return {"type": "scale_up", "target": self._suspected}
        return {"type": "restart_service", "target": self._suspected}
