"""Realistic-ish ice cream for the aligned crop: shaded + textured + specular scoop and a waffle cone to the border.
Anti-aliased by 3x supersampling. Returns (image, alpha_gt)."""
import numpy as np, cv2
def _noise(S, rng, sigma_px, amp):
    n = rng.standard_normal((S, S)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma_px)
    n /= (n.std() + 1e-6)
    return n * amp
def food(crop, cx, cy, r, col, rng, cone=True, texture=1.0, spec=True, tilt=0.5, ss=3):
    """crop float32 HxWx3 (S x S). cx,cy,r in crop fractions. col BGR 0-255. Returns out, alpha"""
    S = crop.shape[0]; SS = int(ss)
    n = S * SS
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    X = xx / SS; Y = yy / SS
    # scoop: a circle slightly squashed
    d = np.sqrt(((X - cx * S) / (r * S)) ** 2 + ((Y - cy * S) / (r * 0.92 * S)) ** 2)
    scoop = (d < 1.0).astype(np.float32)
    # lambert-ish shading with light from upper-left
    nx = (X - cx * S) / (r * S); ny = (Y - cy * S) / (r * 0.92 * S)
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    L = np.array([-0.45, -0.55, 0.70], np.float32); L /= np.linalg.norm(L)
    lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    shade = 0.62 + 0.45 * lam
    base = np.array(col, np.float32)[None, None, :] * shade[:, :, None]
    # texture (bumpy, low freq) + fine grain
    tx = _noise(n, rng, 2.5 * SS, 7.0 * texture) + _noise(n, rng, 0.8 * SS, 3.0 * texture)
    base += tx[:, :, None]
    # darker rim (ambient occlusion)
    rim = np.clip((d - 0.82) / 0.18, 0, 1)
    base *= (1 - 0.25 * rim)[:, :, None]
    if spec:
        sx, sy = cx * S - 0.35 * r * S, cy * S - 0.38 * r * S
        sd = ((X - sx) / (0.22 * r * S)) ** 2 + ((Y - sy) / (0.14 * r * S)) ** 2
        base += (np.exp(-sd) * 70.0)[:, :, None]
    img = base
    alpha = scoop.copy()
    if cone:
        # cone: triangle from below the scoop to the crop border (bottom-right), waffle pattern
        ax, ay = cx * S, cy * S + 0.55 * r * S
        ang = np.deg2rad(55 + tilt * 20)          # direction going down-right
        dirv = np.array([np.cos(ang), np.sin(ang)], np.float32)
        perp = np.array([-dirv[1], dirv[0]], np.float32)
        t = (X - ax) * dirv[0] + (Y - ay) * dirv[1]
        u = (X - ax) * perp[0] + (Y - ay) * perp[1]
        halfw = (0.88 * r * S) * np.clip(1 - t / (1.6 * S), 0, 1)
        cone_m = ((t > 0) & (np.abs(u) < halfw)).astype(np.float32)
        waffle = 0.5 + 0.5 * np.sin(u * 0.55) * np.sin(t * 0.55)
        ccol = np.array([60, 120, 175], np.float32)[None, None, :] * (0.75 + 0.35 * waffle)[:, :, None]
        ccol += tx[:, :, None] * 0.5
        # scoop over cone
        img = np.where(scoop[:, :, None] > 0, base, ccol)
        alpha = np.maximum(scoop, cone_m)
    img_s = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(alpha, (S, S), interpolation=cv2.INTER_AREA)
    # slight lens softness
    img_s = cv2.GaussianBlur(img_s, (0, 0), 0.6 * S / 256.0)
    out = crop * (1 - a_s[:, :, None]) + img_s * a_s[:, :, None]
    return out, a_s
