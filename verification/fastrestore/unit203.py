"""v11.2.203 (SDOS-203): the ArcFace skip and the encrypted saved inputs. Stand-in faces and files, no models, no user media.
usage: unit203.py PACKDIR WORKDIR"""
import sys, types, os, json, shutil
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
PACK, WORK = sys.argv[1], sys.argv[2]
os.environ["PHOENIX_PERSISTENT_ROOT"] = WORK + "/jobs"; os.environ["PHOENIX_PERSISTENT_JOB_STATE_DIR"] = WORK + "/jobs"
os.environ["PHOENIX_PERSISTENT_OUTPUT_DIR"] = WORK + "/out"
sys.path.insert(0, PACK)
import numpy as np, cv2, core_pipeline as cp, swap_engine as E
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
rng = np.random.default_rng(7)
def unit(v): return v / np.linalg.norm(v)
ea, eb = unit(rng.normal(size=512)), unit(rng.normal(size=512))
class F:
    def __init__(s, b, emb):
        s.bbox = np.array(b, np.float32); x1, y1, x2, y2 = b; w = x2 - x1
        s.kps = np.array([[x1+.3*w, y1+.4*w], [x1+.7*w, y1+.4*w], [x1+.5*w, y1+.6*w], [x1+.35*w, y1+.8*w], [x1+.65*w, y1+.8*w]], np.float32)
        s.normed_embedding = None; s.embedding = None; s._emb_img = "img"; s._emb_kps = s.kps
        s._true = emb; s.det_score = 0.9; s.landmark_2d_106 = None; s.gender = 1; s.sex = "M"
def run(xa, xb, n=30, third=False):
    """Two people (A at xa, B at xb). Each call: _arc_reuse, then 'ArcFace' for the faces left without an embedding."""
    T = E.MultiFaceTracker([0, 1]); skipped = embedded = wrong = 0
    for k in range(n):
        fa, fb = F([xa, 300, xa + 200, 500], ea), F([xb, 300, xb + 200, 500], eb)
        faces = [fa, fb] + ([F([500, 40, 620, 160], unit(rng.normal(size=512)))] if third else [])
        r = cp._arc_reuse(faces, T, {0: 0, 1: 1})
        for f in faces:
            if f.normed_embedding is None:
                f.normed_embedding = f.embedding = f._true; embedded += 1
            else:
                skipped += 1
        res = T.assign(faces, {})
        if k >= 5:
            wrong += int(res.get(0) is not fa or res.get(1) is not fb)
    return skipped, embedded, wrong
cp._ARC_STATS.update(reused=0, frames=0)
s1, e1, w1 = run(150, 900)
check("two people far apart: ArcFace skipped on about 2 calls in 3, every face still given to its own person",
      s1 >= 30 and w1 == 0, "%d faces skipped, %d embedded, %d wrong assignments" % (s1, e1, w1))
s2, e2, w2 = run(400, 560)
check("two people close (a kiss, a hug): ArcFace runs on every face", s2 == 0 and w2 == 0, "%d skipped" % s2)
s3, e3, w3 = run(150, 900, third=True)
check("a third face in the picture: ArcFace runs on every face", s3 == 0, "%d skipped" % s3)
E._P["arc_skip"] = 0; s4, _, _ = run(150, 900); E._P["arc_skip"] = 1
check("arc_skip 0: every face embedded (v11.2.92)", s4 == 0, "%d skipped" % s4)
# a person jumps to the other one's place (not possible without passing close by - but a detector box can): no reuse
T = E.MultiFaceTracker([0, 1])
for k in range(10):
    fs = [F([150, 300, 350, 500], ea), F([900, 300, 1100, 500], eb)]; cp._arc_reuse(fs, T, {0: 0, 1: 1})
    for f in fs:
        if f.normed_embedding is None: f.normed_embedding = f.embedding = f._true
    T.assign(fs, {})
fs = [F([600, 300, 800, 500], ea), F([900, 300, 1100, 500], eb)]
r = cp._arc_reuse(fs, T, {0: 0, 1: 1})
check("a face that is not on its person's last box: no reuse", r == 0)

# ---- encryption ----------------------------------------------------------------------------------------------------------
os.makedirs(WORK, exist_ok=True)
big = os.path.join(WORK, "video_plain.bin"); data = os.urandom(9_500_000)          # 3 chunks of 4 MiB
open(big, "wb").write(data)
os.environ["PHOENIX_STORE_KEY"] = "unit-test-secret-0123456789abcdef"
enc, dec = big + ".enc", big + ".dec"
cp._encrypt_file(big, enc); cp._decrypt_file(enc, dec)
raw = open(enc, "rb").read()
check("round trip: the decrypted file is identical (9.5 MB, 3 chunks)", open(dec, "rb").read() == data)
check("the sealed file does not contain the plain bytes", data[1000:1064] not in raw and raw.startswith(cp._STORE_MAGIC))
def must_fail(name, mutate):
    bad = big + ".bad"; open(bad, "wb").write(mutate(raw))
    try:
        cp._decrypt_file(bad, big + ".x"); ok = False
    except Exception:
        ok = True
    check(name, ok)
must_fail("a changed byte is refused", lambda b: b[:5000] + bytes([b[5000] ^ 1]) + b[5001:])
must_fail("a cut file (last chunk missing) is refused", lambda b: b[:len(cp._STORE_MAGIC) + 24 + 4 + (4 << 20) + 16])
must_fail("chunks in another order are refused",
          lambda b: (lambda h, c: h + c[1] + c[0] + b"".join(c[2:]))(*(lambda hdr: (b[:hdr], (lambda body: [body[i:i + 4 + (4 << 20) + 16] for i in range(0, len(body), 4 + (4 << 20) + 16)])(b[hdr:])))(len(cp._STORE_MAGIC) + 24)))
os.environ["PHOENIX_STORE_KEY"] = "another-secret-0123456789abcdef!!"
try:
    cp._decrypt_file(enc, big + ".y"); wk = False
except Exception:
    wk = True
check("another secret cannot open it", wk)
os.environ["PHOENIX_STORE_KEY"] = "unit-test-secret-0123456789abcdef"
# the job's saved inputs: sealed, no ZIP password, patch frames kept with the job; a restart gets plain copies in /tmp
jid = "UNIT0203"
src = {0: os.path.join(WORK, "src0.jpg"), 1: os.path.join(WORK, "src1.jpg")}
for p in src.values():
    cv2.imwrite(p, (rng.random((64, 64, 3)) * 255).astype(np.uint8))
vp = os.path.join(WORK, "vid.mp4"); open(vp, "wb").write(os.urandom(200_000))
pdir = os.path.join(WORK, "phoenix_patches", "abc"); os.makedirs(pdir, exist_ok=True)
pf, pm = os.path.join(pdir, "frame.jpg"), os.path.join(pdir, "mask.png")
cv2.imwrite(pf, (rng.random((64, 64, 3)) * 255).astype(np.uint8)); cv2.imwrite(pm, np.zeros((64, 64), np.uint8))
cfg = dict(password="zip-pass-123", patches=[dict(image=pf, mask=pm, t=1.0, group="m1")], fps=30)
ok = cp._keep_job_inputs(jid, src, vp, cfg)
d = cp.JOB_STATE_ROOT / "inputs" / jid
names = sorted(os.listdir(d)); meta = json.loads((d / "resume.json").read_text())
check("saved inputs are sealed files (.enc)", ok and all(n.endswith(".enc") for n in names if n != "resume.json") and len(names) == 6,
      str(names))
check("the result-ZIP password is not saved", "zip-pass-123" not in (d / "resume.json").read_text() and meta.get("had_password") is True)
spec = cp._resume_spec(jid)
okr = spec is not None
if okr:
    s_, v_, c_ = spec
    okr = (open(v_, "rb").read() == open(vp, "rb").read() and all(open(s_[k], "rb").read() == open(src[k], "rb").read() for k in src)
           and open(c_["patches"][0]["image"], "rb").read() == open(pf, "rb").read() and "/tmp" in v_ and "inputs" not in v_)
check("a restart decrypts the video, faces and patch frames into /tmp, identical to the originals", okr,
      (v_ if spec else "no spec"))
os.environ.pop("PHOENIX_STORE_KEY")
check("without the secret a sealed job cannot be resumed (and says so), nothing crashes", cp._resume_spec(jid) is None)
jid2 = "UNIT0204"; ok2 = cp._keep_job_inputs(jid2, src, vp, dict(cfg))
d2 = cp.JOB_STATE_ROOT / "inputs" / jid2; names2 = sorted(os.listdir(d2))
check("without the secret: kept as before (plain), still without the ZIP password",
      ok2 and "video.mp4" in names2 and "zip-pass-123" not in (d2 / "resume.json").read_text(), str(names2))
shutil.rmtree(WORK, ignore_errors=True)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
