from env.simulator import SERVICES


class HeuristicAgent:
    """Uses check_logs on alert + high-CPU services, infers mode from log text, applies matching fix."""

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
    def _candidate_services(obs: dict) -> list[str]:
        """Prefer services called out in alerts (true degradation), then high noisy CPU — root is often missed by CPU-only top-2."""
        metrics = obs["metrics"]
        by_cpu = sorted(SERVICES, key=lambda s: metrics[s]["cpu"], reverse=True)
        alert_svcs: list[str] = []
        for a in obs.get("recent_alerts", []) or []:
            if isinstance(a, str) and ":" in a:
                name = a.split(":", 1)[0].strip()
                if name in SERVICES and name not in alert_svcs:
                    alert_svcs.append(name)
        seen: set[str] = set()
        out: list[str] = []
        for s in alert_svcs + by_cpu:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:4]

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
            "v2.4.1",
            "v2.3.9",
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
                "v2.4.1",
                "v2.3.9",
            )
        ):
            return "bad_deploy"
        if any(x in t for x in ("queue depth", "throttling", "latency spike", "dropping connections")):
            return "overloaded"
        if any(x in t for x in ("heap usage", "gc pressure", "memory usage at")):
            return "memory_leak"
        return "crashed"

    def _pick_suspect(self, candidates: list[str]) -> str:
        best, best_score = None, -1
        for svc in self._checked:
            sc = self._score_incident_keywords(self._log_by_service.get(svc, ""))
            if sc > best_score:
                best_score = sc
                best = svc
        if best is not None and best_score > 0:
            return best
        for svc in candidates:
            if svc in self._checked:
                return svc
        return candidates[0] if candidates else SERVICES[0]

    def act(self, obs: dict) -> dict:
        self._capture_previous_logs(obs)
        metrics = obs["metrics"]

        if not self._diagnosed:
            candidates = self._candidate_services(obs)
            for svc in candidates:
                if svc not in self._checked:
                    self._checked.add(svc)
                    self._last_check_target = svc
                    return {"type": "check_logs", "target": svc}

            combined = "\n".join(self._log_by_service.values())
            self._suspected = self._pick_suspect(candidates)
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
