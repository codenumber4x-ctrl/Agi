"""Procedural texture generation (wood, paper bedding, wall paint, fur).

All textures are generated with numpy so the project has no binary-art
dependencies; they are written to ``assets/textures`` on first build.
"""
import os

import imageio.v2 as imageio
import numpy as np

TEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "textures")


def _smooth_noise(rng, h, w, scale):
    """Bilinearly upsampled value noise."""
    gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
    g = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1.001, h)
    xs = np.linspace(0, gw - 1.001, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def _fbm(rng, h, w, octaves=((64, 0.5), (16, 0.3), (4, 0.2))):
    out = np.zeros((h, w))
    for scale, amp in octaves:
        out += amp * _smooth_noise(rng, h, w, scale)
    return out


def wood_planks(rng, size=1024, planks=6):
    h = w = size
    y, x = np.mgrid[0:h, 0:w].astype(float)
    plank_w = w / planks
    pid = (x // plank_w).astype(int)
    offs = rng.random(planks + 1) * 400
    tone = 0.85 + 0.3 * rng.random(planks + 1)
    yy = y + offs[pid]
    warp = 18 * _fbm(rng, h, w)
    grain = 0.5 + 0.5 * np.sin((x % plank_w) * 0.35 + warp * 1.4 + 0.02 * yy)
    fine = _fbm(rng, h, w, ((8, 0.6), (2, 0.4)))
    base = np.array([0.62, 0.45, 0.30])
    dark = np.array([0.42, 0.28, 0.17])
    t = (0.55 * grain + 0.45 * fine)[..., None]
    img = (base * (1 - t) + dark * t) * tone[pid][..., None]
    seam = ((x % plank_w) < 2.0) | (((yy % 700) < 2.0))
    img[seam] *= 0.45
    return np.clip(img, 0, 1)


def light_wood(rng, size=512):
    h = w = size
    y, x = np.mgrid[0:h, 0:w].astype(float)
    warp = 25 * _fbm(rng, h, w)
    rings = 0.5 + 0.5 * np.sin(0.09 * x + warp)
    fine = _fbm(rng, h, w, ((6, 0.7), (2, 0.3)))
    base = np.array([0.86, 0.72, 0.52])
    dark = np.array([0.70, 0.53, 0.33])
    t = (0.6 * rings + 0.4 * fine)[..., None]
    return np.clip(base * (1 - t) + dark * t, 0, 1)


def paper_bedding(rng, size=1024):
    """Shredded paper / aspen-like bedding: cream flakes with soft shadows."""
    h = w = size
    img = np.ones((h, w, 3)) * np.array([0.80, 0.74, 0.62])
    img *= (0.80 + 0.25 * _fbm(rng, h, w, ((48, 0.5), (12, 0.5))))[..., None]
    for _ in range(9000):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        L, Wd = rng.integers(6, 26), rng.integers(2, 6)
        ang = rng.random() * np.pi
        c = np.array([0.93, 0.89, 0.78]) * (0.82 + 0.25 * rng.random())
        if rng.random() < 0.18:
            c = np.array([0.72, 0.60, 0.43]) * (0.9 + 0.2 * rng.random())
        ts = np.arange(-L, L + 1)
        for k in range(-Wd // 2, Wd // 2 + 1):
            px = (cx + ts * np.cos(ang) - k * np.sin(ang)).astype(int) % w
            py = (cy + ts * np.sin(ang) + k * np.cos(ang)).astype(int) % h
            img[py, px] = c
            img[(py + 2) % h, (px + 1) % w] *= 0.88  # contact shadow
    return np.clip(img, 0, 1)


def wall_paint(rng, size=512):
    h = w = size
    n = _fbm(rng, h, w, ((32, 0.5), (4, 0.5)))
    img = np.array([0.86, 0.85, 0.80]) * (0.93 + 0.08 * n)[..., None]
    return np.clip(img, 0, 1)


def fur(rng, base_rgb, size=512, guard_rgb=None, streak=0.18):
    """Short directional fur streaks used as skin material texture."""
    h = w = size
    img = np.ones((h, w, 3)) * np.array(base_rgb)
    img *= (0.88 + 0.18 * _fbm(rng, h, w, ((24, 0.6), (6, 0.4))))[..., None]
    guard = np.array(guard_rgb if guard_rgb is not None else base_rgb)
    for _ in range(26000):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        L = rng.integers(3, 9)
        shade = 1.0 + streak * (rng.random() - 0.5) * 2
        c = (guard if rng.random() < 0.25 else np.array(base_rgb)) * shade
        ys = (cy + np.arange(L)) % h
        img[ys, cx] = 0.6 * img[ys, cx] + 0.4 * c
    return np.clip(img, 0, 1)


def _save(name, arr):
    imageio.imwrite(os.path.join(TEX_DIR, name), (arr * 255).astype(np.uint8))


def ensure_textures(force=False):
    os.makedirs(TEX_DIR, exist_ok=True)
    jobs = {
        "floor_wood.png": lambda r: wood_planks(r),
        "light_wood.png": lambda r: light_wood(r),
        "bedding.png": lambda r: paper_bedding(r),
        "wall.png": lambda r: wall_paint(r),
        "fur_white.png": lambda r: fur(r, [0.95, 0.93, 0.88], guard_rgb=[0.99, 0.97, 0.93], streak=0.08),
        "fur_agouti.png": lambda r: fur(r, [0.46, 0.34, 0.24], guard_rgb=[0.18, 0.13, 0.10], streak=0.25),
    }
    for i, (name, fn) in enumerate(jobs.items()):
        path = os.path.join(TEX_DIR, name)
        if force or not os.path.exists(path):
            _save(name, fn(np.random.default_rng(100 + i)))
    return TEX_DIR


if __name__ == "__main__":
    ensure_textures(force=True)
    print("textures written to", TEX_DIR)
