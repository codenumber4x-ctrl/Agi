"""Builds a realistic two-rat home-cage scene for MuJoCo.

The cage follows the dimensions of a typical pet / lab rat enclosure that is
recommended for a pair of adult rats (e.g. 80 x 50 x 60 cm wire cage with a
deep plastic base pan, ~1.25 cm (1/2") bar spacing):

    interior footprint : 80 cm (L) x 50 cm (W)  -> ~4 rat body lengths long
    total height       : 60 cm (10 cm base pan + 50 cm wire top)
    bar spacing        : 1.25 cm, bar diameter 2.5 mm
    volume             : 0.24 m^3 (~8.5 ft^3; >= 2.5 ft^3 per rat guideline)

The rat body is DeepMind's anatomically-based "virtual rodent" (Merel et al.
2020, dm_control), a 0.34 kg Long-Evans rat skeleton with a skinned mesh,
67 joints and 38 position actuators.  Two copies are attached to the scene.

Furnishings:
  * paper-pellet bedding over the pan floor
  * WATER section: 500 ml gravity water bottle hung on the front-right short
    side with a stainless sipper tube that pokes through the bars
  * PELLET section: heavy ceramic food bowl in the back-left corner filled
    with ~40 individually simulated rodent-chow pellets (rigid bodies)
  * wooden hide house, a wooden shelf and a ramp
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
TEX_DIR = os.path.join(ASSETS, "textures")

# ---------------------------------------------------------------- dimensions
CAGE_L = 0.80          # x extent (m)
CAGE_W = 0.50          # y extent
PAN_H = 0.10           # plastic base pan height
CAGE_H = 0.60          # total height
BAR_SPACING = 0.0125   # 1/2 inch, standard for adult rats
BAR_R = 0.00125        # 2.5 mm wire
HX, HY = CAGE_L / 2, CAGE_W / 2

# Locations of the two resource "sections" (used by the behaviour controller).
WATER_SPOUT = np.array([HX - 0.035, 0.12, 0.085])     # sipper ball tip
WATER_DRINK_POS = np.array([HX - 0.13, 0.12])          # where the rat stands
FOOD_BOWL = np.array([-HX + 0.10, -HY + 0.10])          # bowl centre
FOOD_EAT_POS = np.array([-HX + 0.20, -HY + 0.17])      # where the rat stands
HIDE_HOUSE = np.array([-HX + 0.11, HY - 0.10])


# ------------------------------------------------------------------ textures
def _save_png(name, img):
    import imageio.v2 as imageio
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, name)
    tmp = path + f".{os.getpid()}.tmp.png"
    imageio.imwrite(tmp, np.clip(img, 0, 255).astype(np.uint8))
    os.replace(tmp, path)          # atomic: safe with parallel render workers
    return path


def _smooth_noise(rng, h, w, scale):
    """Cheap value-noise (bilinear upsampled random grid)."""
    gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
    g = rng.random((gh, gw))
    yi = np.linspace(0, gh - 1.001, h)
    xi = np.linspace(0, gw - 1.001, w)
    y0, x0 = yi.astype(int), xi.astype(int)
    fy, fx = (yi - y0)[:, None], (xi - x0)[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def make_textures(seed=0):
    rng = np.random.default_rng(seed)
    paths = {}

    # Paper / aspen bedding: lots of small light flakes with soft shadows.
    n = 512
    base = 0.80 + 0.10 * _smooth_noise(rng, n, n, 64)
    img = np.ones((n, n, 3)) * base[..., None] * np.array([218, 207, 186])
    for _ in range(9000):
        x, y = rng.integers(0, n, 2)
        rx, ry = rng.integers(2, 7, 2)
        shade = rng.uniform(0.70, 1.08)
        tint = np.array([222, 212, 192]) * shade
        img[max(0, y - ry):y + ry, max(0, x - rx):x + rx] = tint
    img *= (0.90 + 0.10 * _smooth_noise(rng, n, n, 8))[..., None]
    paths["bedding"] = _save_png("bedding.png", img)

    # Light pine wood (hide house, shelf, ramp).
    h, w = 256, 512
    yy = np.linspace(0, 1, h)[:, None]
    grain = np.sin(40 * yy + 6 * _smooth_noise(rng, h, w, 48)) * 0.5 + 0.5
    grain = 0.82 + 0.18 * grain ** 3
    wood = grain[..., None] * np.array([214, 172, 118])
    wood *= (0.92 + 0.08 * _smooth_noise(rng, h, w, 6))[..., None]
    paths["wood"] = _save_png("wood.png", wood)

    # Room floor: darker oak planks.
    h = w = 512
    planks = np.zeros((h, w, 3))
    pw = 64
    for i in range(w // pw):
        tone = rng.uniform(0.75, 1.0)
        g = _smooth_noise(rng, h, pw, 24)
        stripes = 0.88 + 0.12 * np.sin(np.linspace(0, 40, pw)[None, :] + 3 * g)
        planks[:, i * pw:(i + 1) * pw] = (tone * stripes)[..., None] * np.array([112, 80, 55])
        planks[:, i * pw:i * pw + 2] *= 0.55
    paths["floor"] = _save_png("floor.png", planks)

    # Painted plaster wall.
    wall = (0.93 + 0.07 * _smooth_noise(rng, 256, 256, 16))[..., None] * np.array([222, 220, 212])
    paths["wall"] = _save_png("wall.png", wall)

    # Ceramic bowl glaze.
    glaze = (0.95 + 0.05 * _smooth_noise(rng, 128, 128, 8))[..., None] * np.array([70, 110, 160])
    paths["glaze"] = _save_png("glaze.png", glaze)

    # Rodent chow pellet (speckled brown / grey-green).
    pel = np.ones((64, 64, 3)) * np.array([118, 96, 66])
    pel *= (0.75 + 0.35 * rng.random((64, 64)))[..., None]
    paths["pellet"] = _save_png("pellet.png", pel)
    return paths


# --------------------------------------------------------------- helpers
def _box(body, name, size, pos, mat=None, rgba=None, collide=True, group=0, euler=None):
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, group=group)
    if euler is not None:
        g.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
        g.alt.euler = euler
    if mat:
        g.material = mat
    if rgba is not None:
        g.rgba = rgba
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


def _cyl(body, name, r, half, pos, mat=None, rgba=None, collide=False, fromto=None,
         gtype=mujoco.mjtGeom.mjGEOM_CYLINDER, group=0):
    if fromto is not None:
        g = body.add_geom(name=name, type=gtype, size=[r, 0, 0], fromto=fromto, group=group)
    else:
        g = body.add_geom(name=name, type=gtype, size=[r, half, 0], pos=pos, group=group)
    if mat:
        g.material = mat
    if rgba is not None:
        g.rgba = rgba
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


# ------------------------------------------------------------------ cage
def add_materials(spec, tex):
    def texture(name, path, ttype=mujoco.mjtTexture.mjTEXTURE_2D):
        t = spec.add_texture(name=name, type=ttype)
        t.file = path
        return t

    texture("bedding", tex["bedding"])
    texture("wood", tex["wood"])
    texture("floor", tex["floor"])
    texture("wall", tex["wall"])
    texture("glaze", tex["glaze"], mujoco.mjtTexture.mjTEXTURE_CUBE)
    texture("pellet", tex["pellet"], mujoco.mjtTexture.mjTEXTURE_CUBE)

    def material(name, **kw):
        m = spec.add_material(name=name)
        textures = kw.pop("texture", None)
        if textures:
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = textures
        for k, v in kw.items():
            setattr(m, k, v)
        return m

    material("bedding", texture="bedding", texrepeat=[3, 2], specular=0.05, shininess=0.0)
    material("wood", texture="wood", texrepeat=[1, 1], texuniform=False, specular=0.15, shininess=0.2)
    material("floor", texture="floor", texrepeat=[3, 3], specular=0.25, shininess=0.4, reflectance=0.0)
    material("wall", texture="wall", texrepeat=[2, 2], specular=0.05)
    material("glaze", texture="glaze", specular=0.9, shininess=0.9, reflectance=0.05)
    material("pellet", texture="pellet", specular=0.05, shininess=0.0)
    material("pan", rgba=[0.86, 0.88, 0.90, 0.55], specular=0.6, shininess=0.6)
    material("pan_rim", rgba=[0.18, 0.20, 0.24, 1], specular=0.4, shininess=0.5)
    material("wire", rgba=[0.80, 0.81, 0.83, 1], specular=1.0, shininess=0.9, reflectance=0.1)
    material("steel", rgba=[0.75, 0.77, 0.80, 1], specular=1.0, shininess=1.0, reflectance=0.2)
    material("bottle", rgba=[0.85, 0.92, 0.98, 0.28], specular=1.0, shininess=1.0)
    material("water", rgba=[0.35, 0.62, 0.95, 0.45], specular=1.0, shininess=1.0)
    material("cap", rgba=[0.10, 0.45, 0.75, 1], specular=0.5, shininess=0.6)


def add_room(spec):
    wb = spec.worldbody
    wb.add_geom(name="room_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                pos=[0, 0, -0.004], material="floor")
    wb.add_geom(name="wall_back", type=mujoco.mjtGeom.mjGEOM_BOX, size=[3, 0.02, 1.3],
                pos=[0, 1.4, 1.2], material="wall", contype=0, conaffinity=0)
    wb.add_geom(name="wall_left", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 3, 1.3],
                pos=[-1.6, 0, 1.2], material="wall", contype=0, conaffinity=0)
    wb.add_geom(name="wall_right", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 3, 1.3],
                pos=[1.8, 0, 1.2], material="wall", contype=0, conaffinity=0)
    wb.add_geom(name="baseboard", type=mujoco.mjtGeom.mjGEOM_BOX, size=[3, 0.03, 0.05],
                pos=[0, 1.37, 0.05], rgba=[0.95, 0.95, 0.93, 1], contype=0, conaffinity=0)

    # Lighting: warm ceiling key light + soft fill + window light.
    # Key light sits just below the wire roof (like light pouring in from a
    # ceiling lamp, but without the roof grid shadowing every surface).
    wb.add_light(name="key", pos=[0.10, -0.12, CAGE_H - 0.03], dir=[-0.1, 0.12, -1],
                 castshadow=True, diffuse=[0.42, 0.41, 0.38], specular=[0.2, 0.2, 0.2],
                 cutoff=75, exponent=1, attenuation=[1, 0, 0])
    wb.add_light(name="ceiling", pos=[0.25, -0.35, 1.8], dir=[-0.15, 0.22, -1], castshadow=False,
                 diffuse=[0.22, 0.21, 0.20], specular=[0.1, 0.1, 0.1], cutoff=60, exponent=1)
    wb.add_light(name="fill", pos=[-1.2, -1.5, 1.2], dir=[0.6, 0.7, -0.5], castshadow=False,
                 diffuse=[0.24, 0.25, 0.28], specular=[0.03, 0.03, 0.03])
    wb.add_light(name="window", pos=[1.5, 0.8, 1.4], dir=[-0.8, -0.5, -0.6], castshadow=False,
                 diffuse=[0.25, 0.27, 0.30], specular=[0.0, 0.0, 0.0])


def add_cage(spec, rng):
    wb = spec.worldbody
    cage = wb.add_body(name="cage", pos=[0, 0, 0])
    t = 0.005  # pan wall thickness

    # --- plastic base pan (floor + 4 walls) with dark rim
    _box(cage, "pan_floor", [HX + t, HY + t, 0.004], [0, 0, -0.004], mat="pan_rim")
    _box(cage, "bedding", [HX, HY, 0.0005], [0, 0, 0.0005], mat="bedding")
    for name, size, pos in [
        ("pan_xp", [t, HY + t, PAN_H / 2], [HX + t, 0, PAN_H / 2]),
        ("pan_xn", [t, HY + t, PAN_H / 2], [-HX - t, 0, PAN_H / 2]),
        ("pan_yp", [HX + t, t, PAN_H / 2], [0, HY + t, PAN_H / 2]),
        ("pan_yn", [HX + t, t, PAN_H / 2], [0, -HY - t, PAN_H / 2]),
    ]:
        _box(cage, name, size, pos, mat="pan")
    for name, size, pos in [
        ("rim_xp", [0.008, HY + 0.012, 0.006], [HX + t, 0, PAN_H]),
        ("rim_xn", [0.008, HY + 0.012, 0.006], [-HX - t, 0, PAN_H]),
        ("rim_yp", [HX + 0.012, 0.008, 0.006], [0, HY + t, PAN_H]),
        ("rim_yn", [HX + 0.012, 0.008, 0.006], [0, -HY - t, PAN_H]),
    ]:
        _box(cage, name, size, pos, mat="pan_rim", collide=False)

    # --- invisible collision walls for the wire section (cheap, robust)
    wire_h = (CAGE_H - PAN_H) / 2
    zc = PAN_H + wire_h
    for name, size, pos in [
        ("wire_col_xp", [0.002, HY, wire_h], [HX + 0.002, 0, zc]),
        ("wire_col_xn", [0.002, HY, wire_h], [-HX - 0.002, 0, zc]),
        ("wire_col_yp", [HX, 0.002, wire_h], [0, HY + 0.002, zc]),
        ("wire_col_yn", [HX, 0.002, wire_h], [0, -HY - 0.002, zc]),
        ("wire_col_top", [HX, HY, 0.002], [0, 0, CAGE_H + 0.002]),
    ]:
        _box(cage, name, size, pos, rgba=[0, 0, 0, 0], group=3)

    # --- visual wire bars (vertical) on all four sides
    z0, z1 = PAN_H, CAGE_H
    k = 0
    for x in np.arange(-HX, HX + 1e-6, BAR_SPACING):
        for y in (-HY - 0.002, HY + 0.002):
            _cyl(cage, f"bar{k}", BAR_R, 0, None, mat="wire", fromto=[x, y, z0, x, y, z1]); k += 1
    for y in np.arange(-HY + BAR_SPACING, HY - 1e-6, BAR_SPACING):
        for x in (-HX - 0.002, HX + 0.002):
            _cyl(cage, f"bar{k}", BAR_R, 0, None, mat="wire", fromto=[x, y, z0, x, y, z1]); k += 1
    # roof bars (along y) + thicker frame rails
    for x in np.arange(-HX, HX + 1e-6, BAR_SPACING * 2):
        _cyl(cage, f"bar{k}", BAR_R, 0, None, mat="wire", fromto=[x, -HY, CAGE_H, x, HY, CAGE_H]); k += 1
    for z in (PAN_H + 0.004, PAN_H + 0.25, CAGE_H):
        for a, b in [((-HX, -HY), (HX, -HY)), ((HX, -HY), (HX, HY)),
                     ((HX, HY), (-HX, HY)), ((-HX, HY), (-HX, -HY))]:
            _cyl(cage, f"rail{k}", 0.0028, 0, None, mat="wire",
                 fromto=[a[0], a[1], z, b[0], b[1], z]); k += 1
    for z in np.arange(-HY + 0.10, HY - 0.05, 0.10):
        _cyl(cage, f"roofrail{k}", 0.0022, 0, None, mat="wire",
             fromto=[-HX, z, CAGE_H + 0.002, HX, z, CAGE_H + 0.002]); k += 1
    # door frame on the front (y = -HY)
    for x in (-0.12, 0.12):
        _cyl(cage, f"door{k}", 0.003, 0, None, mat="wire",
             fromto=[x, -HY - 0.004, PAN_H + 0.06, x, -HY - 0.004, CAGE_H - 0.05]); k += 1
    for z in (PAN_H + 0.06, CAGE_H - 0.05):
        _cyl(cage, f"door{k}", 0.003, 0, None, mat="wire",
             fromto=[-0.12, -HY - 0.004, z, 0.12, -HY - 0.004, z]); k += 1
    _box(cage, "latch", [0.012, 0.004, 0.006], [0.0, -HY - 0.008, CAGE_H - 0.05],
         mat="steel", collide=False)

    # --- loose paper bedding pieces (visual) for texture/depth
    for i in range(450):
        x = rng.uniform(-HX + 0.01, HX - 0.01)
        y = rng.uniform(-HY + 0.01, HY - 0.01)
        s = rng.uniform(0.002, 0.0045)
        shade = rng.uniform(0.62, 0.86)
        g = _box(cage, f"flake{i}", [s, s * rng.uniform(0.4, 1.0), 0.0012],
                 [x, y, 0.0008 + rng.uniform(0, 0.0015)],
                 rgba=[0.90 * shade, 0.84 * shade, 0.72 * shade, 1], collide=False,
                 euler=[rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, np.pi)])

    add_water_section(cage)
    add_food_section(spec, cage, rng)
    add_furniture(cage)
    return cage


def add_water_section(cage):
    """500 ml gravity water bottle hung outside the right wall, sipper inside."""
    bx, by = HX + 0.045, WATER_SPOUT[1]
    bz = 0.30
    _cyl(cage, "bottle", 0.036, 0.095, [bx, by, bz], mat="bottle")
    _cyl(cage, "water", 0.033, 0.065, [bx, by, bz - 0.025], mat="water")
    _cyl(cage, "bottle_cap", 0.024, 0.012, [bx, by, bz - 0.105], mat="cap")
    cage.add_geom(name="bottle_top", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  size=[0.036, 0.036, 0.015], pos=[bx, by, bz + 0.095], material="bottle",
                  contype=0, conaffinity=0)
    # spring holder hooked over the bars
    for dz in (0.05, -0.04):
        _cyl(cage, f"bottle_strap{dz}", 0.0016, 0, None, mat="steel",
             fromto=[HX + 0.004, by - 0.04, bz + dz, bx, by - 0.037, bz + dz])
        _cyl(cage, f"bottle_strap2{dz}", 0.0016, 0, None, mat="steel",
             fromto=[HX + 0.004, by + 0.04, bz + dz, bx, by + 0.037, bz + dz])
    # stainless sipper tube angled down through the bars into the cage
    top = [bx - 0.004, by, bz - 0.115]
    tip = list(WATER_SPOUT)
    _cyl(cage, "sipper", 0.0042, 0, None, mat="steel", fromto=top + tip,
         gtype=mujoco.mjtGeom.mjGEOM_CAPSULE)
    cage.add_geom(name="sipper_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.0035, 0, 0],
                  pos=tip, material="steel", contype=0, conaffinity=0)
    cage.add_site(name="water_spout", pos=tip, size=[0.004, 0, 0], rgba=[0, 0, 1, 0])


def add_food_section(spec, cage, rng):
    """Heavy ceramic bowl filled with simulated rodent-chow pellets."""
    cx, cy = FOOD_BOWL
    R, H, t = 0.058, 0.032, 0.007
    _cyl(cage, "bowl_base", R + 0.004, 0.005, [cx, cy, 0.005], mat="glaze", collide=True)
    nseg = 28
    for i in range(nseg):
        a = 2 * np.pi * i / nseg
        _box(cage, f"bowl_wall{i}", [t / 2, R * np.pi / nseg + 0.002, H / 2],
             [cx + R * np.cos(a), cy + R * np.sin(a), 0.010 + H / 2],
             mat="glaze", euler=[0, -0.18, a])
    cage.add_geom(name="bowl_lip", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  size=[R + 0.006, R + 0.006, 0.004], pos=[cx, cy, 0.010 + H],
                  rgba=[0, 0, 0, 0], contype=0, conaffinity=0)
    cage.add_site(name="food_bowl", pos=[cx, cy, 0.03], size=[0.004, 0, 0], rgba=[0, 0, 0, 0])

    # pellets: ~1.5 cm long extruded chow blocks, each a free rigid body
    k = 0
    for layer in range(3):
        for i in range(14):
            r = np.sqrt(rng.uniform(0, 1)) * (R - 0.022)
            a = rng.uniform(0, 2 * np.pi)
            p = spec.worldbody.add_body(name=f"pellet{k}",
                                        pos=[cx + r * np.cos(a), cy + r * np.sin(a),
                                             0.02 + 0.014 * layer + rng.uniform(0, 0.004)])
            p.add_freejoint()
            q = np.zeros(4)
            mujoco.mju_euler2Quat(q, rng.uniform(-np.pi, np.pi, 3), "xyz")
            p.quat = q
            p.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0055, 0.0055, 0],
                       material="pellet", mass=0.0012, friction=[0.9, 0.01, 0.001],
                       condim=4)
            k += 1
    # a couple of pellets the rats have dragged out onto the bedding
    for (dx, dy) in [(0.09, 0.03), (0.11, 0.07), (0.05, 0.10)]:
        p = spec.worldbody.add_body(name=f"pellet{k}", pos=[cx + dx, cy + dy, 0.008])
        p.add_freejoint()
        p.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0055, 0.0055, 0],
                   material="pellet", mass=0.0012, condim=4)
        k += 1


def add_furniture(cage):
    # Wooden hide house in the back-left corner (open door facing +x).
    hx, hy = HIDE_HOUSE
    L, W, H, t = 0.085, 0.07, 0.085, 0.006
    _box(cage, "house_back", [t, W, H / 2], [hx - L, hy, H / 2], mat="wood")
    _box(cage, "house_side_a", [L, t, H / 2], [hx, hy - W, H / 2], mat="wood")
    _box(cage, "house_side_b", [L, t, H / 2], [hx, hy + W, H / 2], mat="wood")
    _box(cage, "house_front_a", [t, 0.022, H / 2], [hx + L, hy - W + 0.022, H / 2], mat="wood")
    _box(cage, "house_front_b", [t, 0.022, H / 2], [hx + L, hy + W - 0.022, H / 2], mat="wood")
    _box(cage, "house_lintel", [t, W - 0.044, 0.014], [hx + L, hy, H - 0.014], mat="wood")
    _box(cage, "house_roof", [L + 0.012, W + 0.012, t], [hx, hy, H + t], mat="wood")

    # Wooden shelf on the back wall + ramp up to it (classic multi-level cage).
    sz = PAN_H + 0.17
    _box(cage, "shelf", [0.12, 0.08, 0.006], [0.20, HY - 0.08, sz], mat="wood")
    for dx in (-0.10, 0.10):
        _cyl(cage, f"shelf_hook{dx}", 0.002, 0, None, mat="steel",
             fromto=[0.20 + dx, HY, sz + 0.01, 0.20 + dx, HY - 0.02, sz + 0.006])
    # ramp from the floor up to the shelf (along -x direction)
    rl = 0.17
    ang = np.arctan2(sz, 0.30)
    rx = 0.08 - 0.5 * 0.30 * 0.95
    _box(cage, "ramp", [rl, 0.035, 0.004], [0.08 - 0.15, HY - 0.05, sz / 2],
         mat="wood", euler=[0, -ang, 0])
    for i in range(9):
        s = -rl + (i + 0.5) * 2 * rl / 9
        _box(cage, f"ramp_rung{i}", [0.003, 0.035, 0.003],
             [0.08 - 0.15 + s * np.cos(ang), HY - 0.05, sz / 2 + s * np.sin(ang) + 0.006],
             mat="wood", collide=False, euler=[0, -ang, 0])
    del rx


# ------------------------------------------------------------------ rats
RAT_COATS = {
    # albino (Sprague-Dawley / Wistar look)
    "rat1": [0.95, 0.92, 0.86, 1.0],
    # agouti / brown hooded (Long-Evans look)
    "rat2": [0.34, 0.26, 0.19, 1.0],
}
RAT_EYES = {"rat1": [0.75, 0.08, 0.12, 1], "rat2": [0.02, 0.02, 0.02, 1]}


def add_rat(spec, name, pos, yaw):
    rat = mujoco.MjSpec.from_file(os.path.join(ASSETS, "rodent.xml"))
    # drop the walker's own overhead tracking light (it would shine through the roof)
    for light in list(rat.lights):
        rat.delete(light)
    frame = spec.worldbody.add_frame(pos=pos)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, [0, 0, yaw], "xyz")
    frame.quat = q
    torso = frame.attach_body(rat.body("torso"), f"{name}/", "")
    torso.add_freejoint(name=f"{name}/root")
    # Eyes: small glossy spheres on the skull (the skin mesh has no eyeballs).
    skull = spec.body(f"{name}/skull")
    for side, s in (("L", 1), ("R", -1)):
        skull.add_geom(name=f"{name}/eye_{side}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[0.0021, 0, 0], pos=[0.0165, s * 0.0098, 0.0075],
                       rgba=RAT_EYES[name], contype=0, conaffinity=0, group=1)
    return torso


def build(seed=0, rat_poses=None, regen_textures=True):
    rng = np.random.default_rng(seed)
    names = ("bedding", "wood", "floor", "wall", "glaze", "pellet")
    if regen_textures:
        tex = make_textures(seed)
    else:
        tex = {n: os.path.join(TEX_DIR, n + ".png") for n in names}
    spec = mujoco.MjSpec()
    spec.modelname = "rat_home_cage"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.map.znear = 0.002
    spec.visual.headlight.ambient = [0.20, 0.20, 0.21]
    spec.visual.headlight.diffuse = [0.18, 0.18, 0.18]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.visual.rgba.haze = [0.85, 0.85, 0.85, 1]
    spec.stat.extent = 1.0
    spec.stat.center = [0, 0, 0.2]

    add_materials(spec, tex)
    add_room(spec)
    add_cage(spec, rng)

    if rat_poses is None:
        rat_poses = {"rat1": ([0.10, -0.08, 0.0], 2.6), "rat2": ([-0.05, 0.08, 0.0], -0.4)}
    for name, (pos, yaw) in rat_poses.items():
        add_rat(spec, name, pos, yaw)

    # Cameras
    wb = spec.worldbody
    def cam(name, pos, target, fovy=45):
        c = wb.add_camera(name=name, pos=pos, fovy=fovy)
        c.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        c.targetbody = target
        return c
    wb.add_body(name="cam_target", pos=[0, 0, 0.08])
    wb.add_body(name="cam_target_floor", pos=[0, 0, 0.03])
    cam("overview", [0.55, -1.05, 0.75], "cam_target", 42)
    cam("front", [0.0, -0.95, 0.22], "cam_target_floor", 42)
    cam("water_cam", [0.18, -0.30, 0.20], "cam_target_floor", 50)
    wb.add_camera(name="top", pos=[0, 0, 1.25], fovy=40)
    return spec


LIMB_KEYS = ("hip", "knee", "ankle", "shoulder", "elbow", "wrist", "scapula")


def strengthen_rat_actuators(model, stiffness=3.0, tau=0.015, force=1.5):
    """The stock rodent servos are tuned for a learned neural-network policy
    (soft, slow).  For a hand-designed CPG gait we make the limb position
    servos 3x stiffer, faster (15 ms activation filter) and allow 1.5x the
    torque, which keeps the joints tracking the gait within ~25 ms."""
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if name and "/" in name and any(k in name.split("/")[-1] for k in LIMB_KEYS):
            model.actuator_gainprm[a, 0] *= stiffness
            model.actuator_biasprm[a, :2] *= stiffness
            model.actuator_dynprm[a, 0] = tau
            model.actuator_forcerange[a] *= force


def compile_scene(seed=0, rat_poses=None, regen_textures=True):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = build(seed, rat_poses, regen_textures)
        model = spec.compile()
    strengthen_rat_actuators(model)
    # Rat skins: coat colours.
    for name, rgba in RAT_COATS.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SKIN, f"{name}/skin")
        if sid >= 0:
            model.skin_rgba[sid] = rgba
    return spec, model


if __name__ == "__main__":
    spec, model = compile_scene()
    print(f"bodies={model.nbody} geoms={model.ngeom} dofs={model.nv} actuators={model.nu} "
          f"skins={model.nskin}")
    out = os.path.join(HERE, "rat_cage_scene.xml")
    with open(out, "w") as f:
        f.write(spec.to_xml())
    print("wrote", out)
