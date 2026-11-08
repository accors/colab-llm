# colab-llm: watchdog. Runs inside the GPU session's kernel via
# `colab exec --timeout 0` (bin/colab-llm prepends common.py) and never
# returns while the server is in use.
#
# Why it lives in the kernel: the Colab CLI has no keep-alive daemon any more;
# the backend keeps a VM only while its kernel is executing. This loop is that
# execution. It also decides when the VM should go away:
#   idle         no tokens processed for IDLE_MINUTES
#   max          session older than MAX_HOURS (0 = no cap)
#   server_dead  llama-server stopped answering /health
#   stop         STOP file created by `colab-llm restart` (VM stays up)
# State goes to state/watchdog.json every POLL seconds; the supervisor on the
# NUC reads it over ssh and calls `colab stop` for idle/max/server_dead.
# If the NUC side is gone, the loop still ends and the kernel goes idle, so
# the backend reclaims the VM after its own idle timeout.

import urllib.request

STATE = f"{APP}/state/watchdog.json"
STOP = f"{APP}/state/STOP"
COUNTERS = ("prompt_tokens_total", "tokens_predicted_total")
GAUGES = ("requests_processing", "requests_deferred")


def read_metrics(port, key):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        text = r.read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            name, _, value = line.partition(" ")
            try:
                values[name[len("llamacpp:"):]] = float(value)
            except ValueError:
                pass
    return values


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def write_state(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE)


def main():
    port = int(env("REMOTE_PORT", "8000"))
    idle_s = float(env("IDLE_MINUTES", "15")) * 60
    max_s = float(env("MAX_HOURS", "8")) * 3600
    poll = float(env("POLL", "30"))
    with open(f"{APP}/api_key") as f:
        key = f.read().strip().splitlines()[0]
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    if os.path.exists(STOP):
        os.remove(STOP)

    now = time.time()
    state = {"started": now, "last_activity": now, "heartbeat": now, "exited": False}
    try:  # keep the idle clock and session age across a watchdog relaunch
        with open(STATE) as f:
            prev = json.load(f)
        if not prev.get("exited") or prev.get("reason") == "stop":
            state["started"] = prev.get("started", now)
            state["last_activity"] = max(prev.get("last_activity", now), now - idle_s / 2)
    except (OSError, ValueError):
        pass

    last_counts, bad_health, last_print = None, 0, 0.0
    log(f"watchdog up: idle {idle_s / 60:.0f} min, max {max_s / 3600:.1f} h, port {port}")
    while True:
        now = time.time()
        reason = None
        if os.path.exists(STOP):
            os.remove(STOP)
            reason = "stop"
        elif healthy(port):
            bad_health = 0
            try:
                m = read_metrics(port, key)
                counts = tuple(m.get(c, 0.0) for c in COUNTERS)
                busy = any(m.get(g, 0.0) > 0 for g in GAUGES)
                if busy or (last_counts is not None and counts != last_counts):
                    state["last_activity"] = now
                last_counts = counts
                state["tokens"] = {c: m.get(c, 0.0) for c in COUNTERS}
            except Exception as e:  # noqa: BLE001 - metrics are best effort
                state["metrics_error"] = str(e)[:200]
        else:
            bad_health += 1
            if bad_health >= 4:
                reason = "server_dead"

        idle_for = now - state["last_activity"]
        if reason is None and idle_for >= idle_s:
            reason = "idle"
        if reason is None and max_s > 0 and now - state["started"] >= max_s:
            reason = "max"

        state.update(heartbeat=now, idle_for=idle_for, age=now - state["started"])
        if reason:
            state.update(exited=True, reason=reason)
            write_state(state)
            log(f"WATCHDOG_EXIT reason={reason} idle_for={idle_for / 60:.1f}min age={state['age'] / 3600:.2f}h")
            return
        write_state(state)
        if now - last_print >= 60:
            log(f"ok | idle {idle_for / 60:.1f}/{idle_s / 60:.0f} min | age {state['age'] / 3600:.2f} h")
            last_print = now
        time.sleep(poll)


finish(main)
