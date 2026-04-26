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
        """Store LOG output for the service we last queried. Be lenient on prefix (Space/Gradio must not drop logs)."""
        raw = obs.get("last_action_result")
        lr = "" if raw is None else (raw if isinstance(raw, str) else str(raw))
        head = lr.lstrip()[:24].upper()
        looks_like_logs = "LOGS:" in lr or head.startswith("LOGS")
        if self._last_check_target and looks_like_logs:
            self._log_by_service[self._last_check_target] = lr
        self._last_check_target = None

    @staticmethod
    def _candidate_services(obs: dict) -> list[str]:
        """Alerts + CPU order first, then any remaining service (all 5 may be log-scanned)."""
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
        for s in SERVICES:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @staticmethod
    def _score_incident_keywords(text: str) -> int:
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
    def _text_suggests_mode(t: str, mode: str) -> bool:
        t = (t or "").lower()
        if mode == "bad_deploy":
            return any(
                x in t
                for x in (
                    "rollback available",
                    "config parse",
                    "startup probe failing",
                    "deploy",
                    "v2.4.1",
                    "v2.3.9",
                )
            )
        if mode == "overloaded":
            return any(x in t for x in ("queue depth", "throttling", "latency spike", "dropping connections"))
        if mode == "memory_leak":
            return any(x in t for x in ("heap usage", "gc pressure", "memory usage at"))
        return any(x in t for x in ("exited unexpectedly", "oom kill", "segfault"))

    @classmethod
    def _infer_best_failure_mode(cls, log_by_service: dict[str, str]) -> str:
        """Scan every stored LOG block + aggregate; first match by severity wins (avoids losing api-gateway hints)."""
        chunks = list(log_by_service.values()) + ["\n".join(log_by_service.values())]
        for mode in ("bad_deploy", "overloaded", "memory_leak", "crashed"):
            for ch in chunks:
                if cls._text_suggests_mode(ch, mode):
                    return mode
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

            self._suspected = self._pick_suspect(candidates)
            self._failure_mode_guess = self._infer_best_failure_mode(self._log_by_service)
            # If we picked a suspect with a strong local log, prefer mode implied by that log
            svc_log = self._log_by_service.get(self._suspected or "", "")
            for mode in ("bad_deploy", "overloaded", "memory_leak", "crashed"):
                if self._text_suggests_mode(svc_log, mode):
                    self._failure_mode_guess = mode
                    break
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
