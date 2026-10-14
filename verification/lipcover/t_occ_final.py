import sys, numpy as np, cv2
sys.path.insert(0,'v71'); import swap_engine as E, config
E.configure(**config.ENGINE_TUNABLES)
exec(open('face_syn.py').read())

def run(seq):
    ref=None; nref=0; keepacc=None; out=[]
    for frm in seq:
        keep=E.occluder_keep(frm, ref=ref, nref=nref)
        if keepacc is None or keepacc.shape!=keep.shape: keepacc=np.ones_like(keep)
        o,c=E._P['occ_open'],E._P['occ_close']
        a=np.where(keep<keepacc,o,c).astype(np.float32)
        keepacc=(keepacc*(1.0-a)+keep*a).astype(np.float32)
        out.append(round(float((1.0-keepacc).mean())*100,1))
        ref,nref=E.update_face_ref(ref,nref,frm,keepacc)
    return out

W=int(E._P['occ_ref_warm'])
def build(sd):
    F=lambda **k: face(seed=sd,**k); D=lambda **k: face(seed=sd,gain=0.55,**k)
    U=lambda **k: face(seed=sd,gain=1.7,**k)
    return [
     ("beard only",            [F(beard=True) for _ in range(28)], [], list(range(W,28))),
     ("glasses only",          [F(glasses=True) for _ in range(28)], [], list(range(W,28))),
     ("mouth open/close",      [F(mouth_open=(i//2)%2==0) for i in range(28)], [], list(range(W,28))),
     ("side shadow",           [F(shadow=True) for _ in range(28)], [], list(range(W,28))),
     ("dark skin, clean",      [F(dark=True) for _ in range(28)], [], list(range(W,28))),
     ("light drops 45% @14",   [F() for _ in range(14)]+[D() for _ in range(14)], [], list(range(W,28))),
     ("light rises 70% @14",   [F() for _ in range(14)]+[U() for _ in range(14)], [], list(range(W,28))),
     ("beard + light drop @14",[F(beard=True) for _ in range(14)]+[D(beard=True) for _ in range(14)], [], list(range(W,28))),
     ("vanilla food @14",      [F() for _ in range(14)]+[F(ice=True) for _ in range(14)], list(range(17,28)), []),
     ("chocolate food @14",    [F() for _ in range(14)]+[F(ice=True,ice_col=(40,55,85)) for _ in range(14)], list(range(17,28)), []),
     ("strawberry food @14",   [F() for _ in range(14)]+[F(ice=True,ice_col=(150,150,235)) for _ in range(14)], list(range(17,28)), []),
     ("spoon @14",             [F() for _ in range(14)]+[F(spoon=True) for _ in range(14)], list(range(17,28)), []),
     ("food, dark skin @14",   [F(dark=True) for _ in range(14)]+[F(dark=True,ice=True) for _ in range(14)], list(range(17,28)), []),
     ("food high at lips @14", [F() for _ in range(14)]+[F(ice=True,ice_y=0.74) for _ in range(14)], list(range(17,28)), []),
     ("BEARD + food @14 (on the beard)",
         [F(beard=True) for _ in range(14)]+[F(beard=True,ice=True,ice_y=0.78) for _ in range(14)],
         list(range(17,28)), []),
     ("food, then light drop @21",
         [F() for _ in range(14)]+[F(ice=True) for _ in range(7)]+[D(ice=True) for _ in range(7)],
         list(range(17,28)), []),
    ]
CUT_MIN, KEEP_MAX = 2.0, 1.0
print(f"occ_ref_warm = {W} frames (nothing is cut while the model forms)\n")
print(f"{'sequence':<36}{'verdict':>9}   per-frame % removed")
nfail=0
for sd in (3,17,42):
    print(f"--- seed {sd} ---")
    for name,seq,must,mustnot in build(sd):
        r=run(seq)
        f=[]
        for i in must:
            if r[i]<CUT_MIN: f.append(f"f{i} missed")
        for i in mustnot:
            if r[i]>KEEP_MAX: f.append(f"f{i} false+ {r[i]}%")
        nfail+=len(f)
        v="OK" if not f else "FAIL"
        print(f"{name:<36}{v:>9}   " + " ".join(f"{x:4.1f}" for x in r[::3]))
        if f: print(f"{'':<36}{'':>9}   -> {f[:4]}")
print(f"\n{'ALL PASS' if nfail==0 else str(nfail)+' FAILURES'}")
