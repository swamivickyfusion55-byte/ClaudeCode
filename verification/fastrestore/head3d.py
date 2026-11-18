"""v11.2.213 (SDOS-213): a real 3D head, turned to any angle, as test footage with a known pose - for the faces turned far away from
the camera (only 10-20 % of the face visible), which no photo can show. The head is the public "Lee Perry-Smith" head scan
(Infinite-Realities, CC BY 3.0; as shipped with the three.js examples: LeePerrySmith.glb + Map-COL.jpg). No user media.

A small z-buffer rasteriser (numpy): per pixel the texture coordinates and a soft light, the texture sampled once per frame
(cv2.remap). Perspective camera; the head turned by yaw (about the vertical axis), pitch, roll.
  Head(path_glb, path_tex)              load
  head.render(yaw, pitch, roll, W, H, centre, height_px, bg)  -> (frame BGR uint8, face-visibility mask, vertex-id buffer)
  head.project(points3d, yaw, pitch, roll, W, H, centre, height_px) -> 2D points (also hidden ones)
Angles in degrees; yaw > 0 turns the face to the viewer's right (the person's left)."""
import struct, json, math
import numpy as np, cv2


def _glb(path):
    b = open(path, "rb").read()
    L = struct.unpack("<I", b[12:16])[0]
    J = json.loads(b[20:20 + L])
    off = 20 + L
    BL = struct.unpack("<I", b[off:off + 4])[0]
    BIN = b[off + 8:off + 8 + BL]

    def acc(i):
        a = J["accessors"][i]; bv = J["bufferViews"][a["bufferView"]]
        dt = {5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(BIN, dt, a["count"] * n, o).reshape(a["count"], n) if n > 1 else np.frombuffer(BIN, dt, a["count"], o)
    p = J["meshes"][0]["primitives"][0]
    return (acc(p["attributes"]["POSITION"]).astype(np.float64), acc(p["attributes"]["NORMAL"]).astype(np.float64),
            acc(p["attributes"]["TEXCOORD_0"]).astype(np.float64), acc(p["indices"]).astype(np.int64).reshape(-1, 3))


def rot(yaw, pitch, roll):
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


class Head:
    def __init__(self, glb, tex, forward=(0, 0, 1), up=(0, 1, 0)):
        P, N, UV, F = _glb(glb)
        self.tex = cv2.imread(tex)
        # model axes -> head axes: x to the head's left as seen from the front... (fixed by the caller's forward / up)
        fw, upv = np.asarray(forward, float), np.asarray(up, float)
        right = np.cross(upv, fw)
        B = np.stack([right, upv, fw])            # rows: head x (viewer's right when facing the camera), y up, z towards camera
        c = (P.max(0) + P.min(0)) / 2
        self.P = (P - c) @ B.T
        self.N = N @ B.T
        self.UV = UV
        self.F = F
        self.h = float(self.P[:, 1].max() - self.P[:, 1].min())

    def _cam(self, R, W, H, centre, height_px, dist=4.0):
        # perspective: camera on +z at dist head-heights, focal so the head is height_px tall at the centre
        X = self.P @ R.T
        D = dist * self.h
        f = height_px * D / self.h
        z = D - X[:, 2]
        u = centre[0] + f * X[:, 0] / z
        v = centre[1] - f * X[:, 1] / z
        return u, v, z, X

    def project(self, pts3d, yaw, pitch, roll, W, H, centre, height_px, dist=4.0):
        R = rot(yaw, pitch, roll)
        X = np.asarray(pts3d, float) @ R.T
        D = dist * self.h; f = height_px * D / self.h; z = D - X[:, 2]
        return np.stack([centre[0] + f * X[:, 0] / z, centre[1] - f * X[:, 1] / z], 1)

    def render(self, yaw, pitch, roll, W, H, centre, height_px, bg, light=(-0.35, 0.45, 0.82), dist=4.0):
        R = rot(yaw, pitch, roll)
        u, v, z, X = self._cam(R, W, H, centre, height_px, dist)
        Nr = self.N @ R.T
        L = np.asarray(light, float); L = L / np.linalg.norm(L)
        shade_v = 0.42 + 0.58 * np.clip(Nr @ L, 0, 1)
        zb = np.full((H, W), np.inf); ub = np.zeros((H, W), np.float32); vb = np.zeros((H, W), np.float32)
        sb = np.zeros((H, W), np.float32); idb = np.full((H, W), -1, np.int64)
        for ti, (a, b, c) in enumerate(self.F):
            xs = np.array([u[a], u[b], u[c]]); ys = np.array([v[a], v[b], v[c]])
            area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
            if area >= 0:                       # back-facing (image y down: front faces are clockwise here)
                continue
            x0, x1 = max(int(math.floor(xs.min())), 0), min(int(math.ceil(xs.max())), W - 1)
            y0, y1 = max(int(math.floor(ys.min())), 0), min(int(math.ceil(ys.max())), H - 1)
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((xs[1] - gx) * (ys[2] - gy) - (xs[2] - gx) * (ys[1] - gy)) / area
            w1 = ((xs[2] - gx) * (ys[0] - gy) - (xs[0] - gx) * (ys[2] - gy)) / area
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not ins.any():
                continue
            zz = w0 * z[a] + w1 * z[b] + w2 * z[c]
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            m = ins & (zz < sub)
            if not m.any():
                continue
            sub[m] = zz[m]
            ub[y0:y1 + 1, x0:x1 + 1][m] = (w0 * self.UV[a, 0] + w1 * self.UV[b, 0] + w2 * self.UV[c, 0])[m]
            vb[y0:y1 + 1, x0:x1 + 1][m] = (w0 * self.UV[a, 1] + w1 * self.UV[b, 1] + w2 * self.UV[c, 1])[m]
            sb[y0:y1 + 1, x0:x1 + 1][m] = (w0 * shade_v[a] + w1 * shade_v[b] + w2 * shade_v[c])[m]
            idb[y0:y1 + 1, x0:x1 + 1][m] = ti
        th, tw = self.tex.shape[:2]
        col = cv2.remap(self.tex, ub * (tw - 1), vb * (th - 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        cov = np.isfinite(zb)
        # soft coverage edge (anti-aliasing): a 1-px blur of the silhouette
        a = cv2.GaussianBlur(cov.astype(np.float32), (3, 3), 0.7)[..., None]
        img = bg.astype(np.float32) * (1 - a) + (col * sb[..., None]) * a
        return np.clip(img, 0, 255).astype(np.uint8), cov, idb
