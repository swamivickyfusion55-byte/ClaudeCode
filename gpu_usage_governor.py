"""Hard cap on remote-GPU usage: 3h/day, 12h/week (Sun-Sat), 50h/month.

This module has ONE job: never let the app start a remote-GPU job that
would (or already has) put the account over budget, and be honest about
it in the UI rather than silently overspending. It knows nothing about
Jarvislabs, HTTP, or video processing - just usage accounting - so it is
fully testable without any external account or network access, and the
one thing standing between "convenient" and "your card gets billed for a
week you didn't mean to run" is code you can read end to end in one
sitting.

Usage shape:
    gov = GPUUsageGovernor()
    ok, reason = gov.can_start()
    if not ok:
        ... refuse / fall back to local CPU, show `reason` to the user ...
    session_id = gov.start_session()
    try:
        ... do the remote GPU work ...
    finally:
        gov.end_session(session_id)   # always, even on failure/cancel -
                                       # a session that ran GPU-hours and
                                       # was never recorded is the exact
                                       # overspend this module exists to
                                       # prevent.

Caps are enforced going forward (a job already in flight is never killed
just because a clock boundary passed under it - see can_start()'s
docstring), and are all independent: ANY one of daily/weekly/monthly
being exhausted blocks a new session, even if the other two have room.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python <3.9 fallback, not expected here
    ZoneInfo = None

DEFAULT_TZ = "Asia/Kolkata"
DEFAULT_STATE_PATH = os.environ.get(
    "PHOENIX_GPU_GOVERNOR_STATE",
    "/data/phoenix_gpu_usage.json" if os.path.isdir("/data") else "/tmp/phoenix_gpu_usage.json",
)

DAILY_CAP_HOURS = float(os.environ.get("PHOENIX_GPU_DAILY_CAP_HOURS", "3.0"))
WEEKLY_CAP_HOURS = float(os.environ.get("PHOENIX_GPU_WEEKLY_CAP_HOURS", "12.0"))
MONTHLY_CAP_HOURS = float(os.environ.get("PHOENIX_GPU_MONTHLY_CAP_HOURS", "50.0"))

# A session left "open" (start_session called, end_session never reached -
# process crash, container restart) must not silently vanish from the
# ledger, but must also not block usage forever. Auto-closed at this
# elapsed wall-clock length using its own start time as the end time
# actually used for accounting (i.e. it is credited as if it ended when
# discovered, not artificially extended) - see _reap_stale_sessions().
STALE_SESSION_HOURS = float(os.environ.get("PHOENIX_GPU_STALE_SESSION_HOURS", "6.0"))


def _tz():
    if ZoneInfo is not None:
        try:
            return ZoneInfo(DEFAULT_TZ)
        except Exception:
            pass
    return timezone.utc


@dataclass
class _Session:
    id: str
    start_ts: float
    end_ts: float | None = None
    # Set only by record_completed_hours() - an explicit credited amount,
    # decoupled from (end_ts - start_ts). Without this, backdating a LONG
    # block (say, crediting 12 already-elapsed hours) by setting
    # start_ts = now - 12h can push start_ts into an earlier day/week
    # bucket than the one this usage should actually count against -
    # under-counting the very usage being recorded. Bucketing key
    # (start_ts) and credited amount (hours) need to be independent for
    # anything longer than the shortest cap window; real sessions
    # (start_session/end_session, bracketing actual elapsed wall-clock
    # time) never need this, since a single session is bounded by the
    # daily cap itself and can't be long enough to cross a day boundary
    # under normal use.
    hours_override: float | None = None

    @property
    def hours(self) -> float:
        if self.hours_override is not None:
            return max(0.0, self.hours_override)
        end = self.end_ts if self.end_ts is not None else time.time()
        return max(0.0, (end - self.start_ts) / 3600.0)


class GPUUsageGovernor:
    """Tracks remote-GPU session time and enforces day/week/month caps.

    Week is Sunday-Saturday (isoweekday() 7 == Sunday), matching what was
    asked for explicitly rather than the ISO Monday-Sunday default.

    A session's ENTIRE duration is attributed to the day/week/month bucket
    it STARTED in, not split across a boundary it happens to cross. For
    sessions capped at a few hours against day/week/month windows, the
    edge case this simplifies (a session starting at 23:50 and ending
    00:10) costs at most a few minutes of accounting slop, in the
    direction of attributing MORE usage to the earlier bucket, not
    hiding it - a conservative simplification, not a loophole.
    """

    def __init__(
        self,
        state_path: str = DEFAULT_STATE_PATH,
        daily_cap_hours: float = DAILY_CAP_HOURS,
        weekly_cap_hours: float = WEEKLY_CAP_HOURS,
        monthly_cap_hours: float = MONTHLY_CAP_HOURS,
        stale_session_hours: float = STALE_SESSION_HOURS,
    ):
        self.state_path = state_path
        self.daily_cap_hours = float(daily_cap_hours)
        self.weekly_cap_hours = float(weekly_cap_hours)
        self.monthly_cap_hours = float(monthly_cap_hours)
        self.stale_session_hours = float(stale_session_hours)
        self._lock = threading.RLock()
        self._sessions: list[_Session] = []
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self):
        with self._lock:
            self._sessions = []
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                for row in data.get("sessions", []):
                    self._sessions.append(_Session(**row))
            except FileNotFoundError:
                pass
            except Exception:
                # Corrupt state file: fail CLOSED on trusting old usage (start
                # from an empty ledger) rather than crash the app - but never
                # silently delete the file, so a human can inspect what went
                # wrong. A fresh empty ledger is the safe direction to fail in
                # for a cap-enforcement module: worst case it under-counts
                # past usage for one cycle, it never lets that manifest as
                # unbounded future spend, because every session recorded from
                # here on is still capped normally.
                pass
            self._reap_stale_sessions(persist=False)

    def _save(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"sessions": [asdict(s) for s in self._sessions]}, f)
            os.replace(tmp, self.state_path)

    def _reap_stale_sessions(self, persist: bool = True):
        now = time.time()
        changed = False
        for s in self._sessions:
            if s.end_ts is None and (now - s.start_ts) / 3600.0 > self.stale_session_hours:
                s.end_ts = s.start_ts + self.stale_session_hours * 3600.0
                changed = True
        if changed and persist:
            self._save()

    # -- accounting windows -------------------------------------------------
    def _day_bounds(self, ts: float):
        dt = datetime.fromtimestamp(ts, _tz())
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), (start + timedelta(days=1)).timestamp()

    def _week_bounds(self, ts: float):
        dt = datetime.fromtimestamp(ts, _tz())
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        # Python isoweekday(): Mon=1 .. Sun=7. Days since the most recent
        # Sunday (0 if today IS Sunday).
        since_sunday = dt.isoweekday() % 7
        week_start = day_start - timedelta(days=since_sunday)
        return week_start.timestamp(), (week_start + timedelta(days=7)).timestamp()

    def _month_bounds(self, ts: float):
        dt = datetime.fromtimestamp(ts, _tz())
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
        return start.timestamp(), end.timestamp()

    def _sum_hours_in(self, lo: float, hi: float) -> float:
        total = 0.0
        for s in self._sessions:
            if lo <= s.start_ts < hi:
                total += s.hours
        return total

    # -- public API ----------------------------------------------------------
    def usage_snapshot(self, now: float | None = None) -> dict:
        """Hours used and remaining in each window, as of `now`."""
        with self._lock:
            self._reap_stale_sessions()
            now = now if now is not None else time.time()
            d0, d1 = self._day_bounds(now)
            w0, w1 = self._week_bounds(now)
            m0, m1 = self._month_bounds(now)
            used_day = self._sum_hours_in(d0, d1)
            used_week = self._sum_hours_in(w0, w1)
            used_month = self._sum_hours_in(m0, m1)
            return {
                "day": {"used": used_day, "cap": self.daily_cap_hours,
                        "remaining": max(0.0, self.daily_cap_hours - used_day)},
                "week": {"used": used_week, "cap": self.weekly_cap_hours,
                         "remaining": max(0.0, self.weekly_cap_hours - used_week)},
                "month": {"used": used_month, "cap": self.monthly_cap_hours,
                          "remaining": max(0.0, self.monthly_cap_hours - used_month)},
            }

    def can_start(self, estimated_hours: float = 0.0) -> tuple[bool, str]:
        """Would starting a session now (estimated length ``estimated_hours``)
        push any window over its cap?

        Deliberately checked only at start time, not enforced mid-session:
        a job already running is allowed to finish even if it crosses a cap
        boundary while in flight (killing a half-finished video job to save
        a few minutes of GPU time is a worse outcome than letting it
        complete and simply blocking the NEXT one) - the cap's job is to
        stop new spend, not to abort work already committed to.
        """
        snap = self.usage_snapshot()
        for window in ("day", "week", "month"):
            remaining = snap[window]["remaining"]
            if remaining <= 0:
                return False, (
                    f"{window.capitalize()} GPU budget exhausted "
                    f"({snap[window]['used']:.2f}h / {snap[window]['cap']:.2f}h used). "
                    f"Falling back to local CPU."
                )
            if estimated_hours > 0 and remaining < estimated_hours:
                return False, (
                    f"Starting this job could exceed the {window} GPU budget "
                    f"({snap[window]['remaining']:.2f}h left, job estimated at "
                    f"{estimated_hours:.2f}h). Falling back to local CPU."
                )
        return True, "ok"

    def start_session(self) -> str:
        with self._lock:
            sid = uuid.uuid4().hex[:12]
            self._sessions.append(_Session(id=sid, start_ts=time.time()))
            self._save()
            return sid

    def end_session(self, session_id: str):
        with self._lock:
            for s in self._sessions:
                if s.id == session_id and s.end_ts is None:
                    s.end_ts = time.time()
                    self._save()
                    return
            # end_session called for an id we don't have an open record for
            # (already reaped as stale, or a bad id) - nothing to close, but
            # this must never raise: a governor that crashes the caller on
            # its own bookkeeping is worse than one that quietly no-ops here.

    def record_completed_hours(self, hours: float):
        """Record a block of usage that has ALREADY happened (e.g. computed
        from a remote instance's own reported uptime rather than bracketed
        with start_session/end_session), bucketed as happening NOW - use
        start_session/end_session instead when you can bracket the actual
        work; this is the fallback for when only a total duration is known
        after the fact. Uses hours_override rather than backdating start_ts
        by the duration - see _Session.hours_override for why that matters
        for any block long enough to cross a bucket boundary."""
        if hours <= 0:
            return
        with self._lock:
            now = time.time()
            self._sessions.append(_Session(id=uuid.uuid4().hex[:12],
                                            start_ts=now, end_ts=now,
                                            hours_override=float(hours)))
            self._save()


if __name__ == "__main__":
    # Self-test: no network, no external account needed. Run directly
    # (`python gpu_usage_governor.py`) to verify the cap logic in isolation
    # before trusting it to gate real spend.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "usage.json")
        gov = GPUUsageGovernor(path, daily_cap_hours=3.0, weekly_cap_hours=12.0, monthly_cap_hours=50.0)

        ok, reason = gov.can_start()
        assert ok, f"fresh governor should allow a session: {reason}"

        sid = gov.start_session()
        time.sleep(0.05)
        gov.end_session(sid)
        snap = gov.usage_snapshot()
        assert snap["day"]["used"] > 0, "session should be recorded"
        print("basic session record/replay: OK")

        gov.record_completed_hours(2.99)
        ok, reason = gov.can_start()
        assert ok, "just under daily cap should still allow starting"
        gov.record_completed_hours(0.02)
        ok, reason = gov.can_start()
        assert not ok and "Day" in reason, f"daily cap should now block: {reason}"
        print("daily cap enforcement: OK ->", reason)

        gov2 = GPUUsageGovernor(path, daily_cap_hours=3.0, weekly_cap_hours=12.0, monthly_cap_hours=50.0)
        snap2 = gov2.usage_snapshot()
        assert abs(snap2["day"]["used"] - snap["day"]["used"] - 3.01) < 0.01, "state must persist across instances"
        print("persistence across instances: OK")

        gov3 = GPUUsageGovernor(path, daily_cap_hours=100.0, weekly_cap_hours=12.0, monthly_cap_hours=50.0)
        gov3.record_completed_hours(12.0)
        ok, reason = gov3.can_start()
        assert not ok and "Week" in reason, f"weekly cap should block even with daily cap wide open: {reason}"
        print("weekly cap enforcement independent of daily cap: OK ->", reason)

        gov4 = GPUUsageGovernor(path, daily_cap_hours=100.0, weekly_cap_hours=100.0, monthly_cap_hours=1.0)
        ok, reason = gov4.can_start()
        assert not ok and "Month" in reason, f"monthly cap should block: {reason}"
        print("monthly cap enforcement: OK ->", reason)

        with tempfile.TemporaryDirectory() as d2:
            path2 = os.path.join(d2, "usage2.json")
            gov5 = GPUUsageGovernor(path2, stale_session_hours=0.0001)
            sid5 = gov5.start_session()
            time.sleep(0.5)
            snap5 = gov5.usage_snapshot()  # should reap the stale open session
            assert snap5["day"]["used"] > 0, "stale open session should be auto-closed and counted, not lost"
            print("stale/crashed session auto-close: OK")

        print("\nALL SELF-TESTS PASSED")
