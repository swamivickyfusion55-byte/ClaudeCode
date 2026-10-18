"""TCP proxy that misbehaves like a phone being minimised: after a random 0.5-3 MB of upstream data it either cuts the connection
or stalls it (no data either way) for a while, then cuts. Usage: flaky.py LISTEN_PORT TARGET_PORT SEED"""
import socket, threading, random, sys, time
L, T, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); rng = random.Random(seed); cuts = [0]; stalls = [0]
def pipe(src, dst, budget, stall):
    sent = 0
    try:
        while True:
            d = src.recv(65536)
            if not d: break
            if budget is not None and sent + len(d) > budget:
                if stall:
                    stalls[0] += 1; time.sleep(stall)          # nothing moves; the client's timeout or our cut ends it
                cuts[0] += 1
                break
            dst.sendall(d); sent += len(d)
    except Exception:
        pass
    for s in (src, dst):
        try: s.shutdown(socket.SHUT_RDWR)
        except Exception: pass
        try: s.close()
        except Exception: pass
def handle(c):
    u = socket.create_connection(("127.0.0.1", T))
    bad = rng.random() < 0.95
    budget = rng.randint(500_000, 3_000_000) if bad else None
    stall = rng.choice([0, 0, 4]) if bad else 0
    threading.Thread(target=pipe, args=(c, u, budget, stall), daemon=True).start()
    threading.Thread(target=pipe, args=(u, c, None, 0), daemon=True).start()
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", L)); s.listen(64)
threading.Thread(target=lambda: [print(f"proxy: {cuts[0]} connections cut, {stalls[0]} stalled", flush=True) or time.sleep(5) for _ in iter(int, 1)], daemon=True).start()
while True:
    c, _ = s.accept(); threading.Thread(target=handle, args=(c,), daemon=True).start()
