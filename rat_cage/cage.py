"""Builds a realistic two-rat enclosure in MuJoCo and loads two virtual rats.

Cage proportions follow a common commercial two-level rat cage
(Critter-Nation-style single unit): 91 cm long x 61 cm deep x 63 cm high
interior, a 12 cm deep plastic base pan, 1.27 cm (1/2") wire bar spacing.
An adult rat in the model is ~22 cm nose-to-rump (+ ~11 cm visible tail),
so the cage is ~4x body length long and ~2.8x deep, which is the same ratio
as typical recommended rat housing (>= ~0.07 m^3 / 2.5 cubic ft per rat).

Contents:
  * paper bedding layer
  * 500 ml gravity water bottle hung on the front bars with a stainless
    sipper tube + ball tip reaching into the cage (WATER SECTION)
  * heavy ceramic food crock filled with lab-block pellets that are
    individually simulated rigid bodies (PELLET SECTION)
  * wooden hideout house, wire half-shelf with liner, wooden chew block
  * two rats: the DeepMind "virtual rodent" (anatomical skeleton of a
    Long-Evans rat, 67 DoF, 38 actuators, skinned body), one albino and
    one agouti, each given a free joint.
"""
import math
import os

import mujoco
import numpy as np

import skin_uv
import textures

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# ---------------------------------------------------------------- dimensions
L, W, H = 0.91, 0.61, 0.63          # interior length (x), depth (y), height (z)
PAN_H = 0.12                        # deep plastic base pan
BED_Z = 0.02                        # top surface of bedding (rats walk here)
BAR_SPACING = 0.0127                # 1/2 inch
BAR_R = 0.0013                      # bar wire radius

# Key locations used by the behaviour controller.
WATER_X = 0.27
SIPPER_TIP = np.array([WATER_X, -W / 2 + 0.045, 0.098])
BOWL_POS = np.array([-0.24, -0.12])
BOWL_R = 0.065
HOUSE_POS = np.array([0.24, 0.16])
HOUSE_SIZE = np.array([0.22, 0.16, 0.12])
CHEW_POS = np.array([-0.05, 0.20])


def _fmt(v):
    return " ".join(f"{x:.5g}" for x in v)


def _cyl_between(p0, p1, r, **attrs):
    a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f'<geom type="capsule" fromto="{_fmt(p0)} {_fmt(p1)}" size="{r}" {a}/>'


def _cam(name, pos, target, fovy):
    """Camera looking from ``pos`` at ``target`` (world z up)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return f'<camera name="{name}" pos="{_fmt(pos)}" xyaxes="{_fmt(right)} {_fmt(up)}" fovy="{fovy}"/>'


def _bars_xml():
    """Wire bar walls, top grid and the frame."""
    g = []
    z0, z1 = PAN_H, H
    hx, hy = L / 2, W / 2
    wire = 'class="wire"'
    # vertical bars on all four sides
    nx = int(L / BAR_SPACING)
    for i in range(nx + 1):
        x = -hx + i * L / nx
        for y in (-hy, hy):
            g.append(f'<geom {wire} fromto="{x:.4f} {y:.4f} {z0} {x:.4f} {y:.4f} {z1}"/>')
    ny = int(W / BAR_SPACING)
    for j in range(1, ny):
        y = -hy + j * W / ny
        for x in (-hx, hx):
            g.append(f'<geom {wire} fromto="{x:.4f} {y:.4f} {z0} {x:.4f} {y:.4f} {z1}"/>')
    # top: bars along y, with cross braces along x
    for i in range(nx + 1):
        x = -hx + i * L / nx
        g.append(f'<geom {wire} fromto="{x:.4f} {-hy} {z1} {x:.4f} {hy} {z1}"/>')
    for y in np.linspace(-hy, hy, 5):
        g.append(f'<geom class="frame_thin" fromto="{-hx} {y:.4f} {z1 + 0.003} {hx} {y:.4f} {z1 + 0.003}"/>')
    # horizontal rails around the walls
    for z in (z0 + 0.004, 0.36, z1):
        g.append(f'<geom class="frame" fromto="{-hx} {-hy} {z} {hx} {-hy} {z}"/>')
        g.append(f'<geom class="frame" fromto="{-hx} {hy} {z} {hx} {hy} {z}"/>')
        g.append(f'<geom class="frame" fromto="{-hx} {-hy} {z} {-hx} {hy} {z}"/>')
        g.append(f'<geom class="frame" fromto="{hx} {-hy} {z} {hx} {hy} {z}"/>')
    # corner posts + legs of the stand
    for sx in (-1, 1):
        for sy in (-1, 1):
            g.append(f'<geom class="post" pos="{sx * (hx + 0.008)} {sy * (hy + 0.008)} {H / 2}" size="0.009 0.009 {H / 2 + 0.008}"/>')
    # front double doors: frame outlines and latches
    for cx in (-0.2, 0.2):
        x0, x1 = cx - 0.17, cx + 0.17
        zb, zt = 0.16, 0.6
        y = -hy - 0.004
        for a, b in (((x0, y, zb), (x1, y, zb)), ((x0, y, zt), (x1, y, zt)),
                     ((x0, y, zb), (x0, y, zt)), ((x1, y, zb), (x1, y, zt))):
            g.append(f'<geom class="frame_thin" fromto="{_fmt(a)} {_fmt(b)}"/>')
        g.append(f'<geom class="latch" pos="{cx + np.sign(cx) * -0.16} {y - 0.006} 0.40" size="0.006 0.006 0.025"/>')
    return "\n      ".join(g)


def _pan_xml():
    hx, hy, t = L / 2 + 0.006, W / 2 + 0.006, 0.006
    g = [
        f'<geom name="pan_floor" class="pan" pos="0 0 {t / 2}" size="{hx} {hy} {t / 2}" contype="0" conaffinity="0"/>',
        f'<geom class="pan" pos="0 {-hy} {PAN_H / 2}" size="{hx + t} {t} {PAN_H / 2}"/>',
        f'<geom class="pan" pos="0 {hy} {PAN_H / 2}" size="{hx + t} {t} {PAN_H / 2}"/>',
        f'<geom class="pan" pos="{-hx} 0 {PAN_H / 2}" size="{t} {hy} {PAN_H / 2}"/>',
        f'<geom class="pan" pos="{hx} 0 {PAN_H / 2}" size="{t} {hy} {PAN_H / 2}"/>',
        # rolled lip of the pan
        f'<geom class="pan_lip" fromto="{-hx} {-hy - 0.004} {PAN_H} {hx} {-hy - 0.004} {PAN_H}"/>',
        f'<geom class="pan_lip" fromto="{-hx} {hy + 0.004} {PAN_H} {hx} {hy + 0.004} {PAN_H}"/>',
        f'<geom class="pan_lip" fromto="{-hx - 0.004} {-hy} {PAN_H} {-hx - 0.004} {hy} {PAN_H}"/>',
        f'<geom class="pan_lip" fromto="{hx + 0.004} {-hy} {PAN_H} {hx + 0.004} {hy} {PAN_H}"/>',
        # bedding layer (collision surface for the rats)
        f'<geom name="bedding" type="box" material="bedding" pos="0 0 {(BED_Z + t) / 2}" '
        f'size="{L / 2} {W / 2} {(BED_Z - t) / 2}" contype="0" conaffinity="0" group="0"/>',
        # collision: a half-space at the bedding surface (cannot be tunnelled through)
        f'<geom name="bedding_col" type="plane" pos="0 0 {BED_Z}" size="{L / 2} {W / 2} 0.01" '
        f'friction="1.2 0.005 0.0001" rgba="0 0 0 0" group="3"/>',
        # invisible collision walls standing in for the bars
        f'<geom class="wallcol" pos="0 {-W / 2 - 0.01} {H / 2}" size="{L / 2} 0.01 {H / 2}"/>',
        f'<geom class="wallcol" pos="0 {W / 2 + 0.01} {H / 2}" size="{L / 2} 0.01 {H / 2}"/>',
        f'<geom class="wallcol" pos="{-L / 2 - 0.01} 0 {H / 2}" size="0.01 {W / 2} {H / 2}"/>',
        f'<geom class="wallcol" pos="{L / 2 + 0.01} 0 {H / 2}" size="0.01 {W / 2} {H / 2}"/>',
    ]
    # loose bedding flakes on top of the bedding surface (visual only)
    rng = np.random.default_rng(7)
    for _ in range(260):
        x = rng.uniform(-L / 2 + 0.01, L / 2 - 0.01)
        y = rng.uniform(-W / 2 + 0.01, W / 2 - 0.01)
        a = rng.uniform(0, math.pi)
        s = rng.uniform(0.004, 0.011)
        col = rng.choice([0, 1, 1, 1])
        mat = "flake_dark" if col == 0 else "flake"
        g.append(f'<geom class="flake" material="{mat}" pos="{x:.4f} {y:.4f} {BED_Z + 0.0008:.4f}" '
                 f'euler="{rng.uniform(-.25, .25):.3f} {rng.uniform(-.25, .25):.3f} {a:.3f}" '
                 f'size="{s:.4f} {s * 0.45:.4f} 0.0008"/>')
    return "\n      ".join(g)


def _water_bottle_xml():
    """500 ml gravity bottle hung outside the front bars; sipper reaches in."""
    x, yb = WATER_X, -W / 2 - 0.045
    zc = 0.30
    tip = SIPPER_TIP
    cap_z = zc - 0.10
    root = np.array([x, yb + 0.004, cap_z - 0.02])
    return f"""
      <body name="water_bottle" pos="0 0 0">
        <geom name="bottle" type="cylinder" pos="{x} {yb} {zc}" size="0.036 0.095" material="bottle_plastic" contype="0" conaffinity="0"/>
        <geom name="bottle_shoulder" type="ellipsoid" pos="{x} {yb} {zc + 0.095}" size="0.036 0.036 0.02" material="bottle_plastic" contype="0" conaffinity="0"/>
        <geom name="water" type="cylinder" pos="{x} {yb} {zc - 0.022}" size="0.0335 0.068" material="water" contype="0" conaffinity="0"/>
        <geom name="water_surface" type="cylinder" pos="{x} {yb} {zc + 0.046}" size="0.0335 0.0006" material="water_top" contype="0" conaffinity="0"/>
        <geom name="label" type="cylinder" pos="{x} {yb} {zc + 0.01}" size="0.0365 0.03" material="label" contype="0" conaffinity="0"/>
        <geom name="cap" type="cylinder" pos="{x} {yb} {cap_z}" size="0.019 0.014" material="cap" contype="0" conaffinity="0"/>
        <geom name="cap_ring" type="cylinder" pos="{x} {yb} {cap_z - 0.016}" size="0.012 0.004" material="steel" contype="0" conaffinity="0"/>
        {_cyl_between(root, tip, 0.0042, material="steel", contype="0", conaffinity="0", name="sipper_tube")}
        <geom name="sipper_ball" type="sphere" pos="{_fmt(tip)}" size="0.0047" material="steel" contype="0" conaffinity="0"/>
        <geom name="drop" type="ellipsoid" pos="{tip[0]} {tip[1] + 0.001} {tip[2] - 0.0062}" size="0.0022 0.0022 0.003" material="water_drop" contype="0" conaffinity="0"/>
        {_cyl_between((x - 0.04, yb, zc + 0.07), (x + 0.04, yb, zc + 0.07), 0.0012, material="steel", contype="0", conaffinity="0")}
        {_cyl_between((x - 0.04, yb, zc - 0.07), (x + 0.04, yb, zc - 0.07), 0.0012, material="steel", contype="0", conaffinity="0")}
        {_cyl_between((x - 0.04, yb, zc + 0.07), (x - 0.04, -W / 2, zc + 0.10), 0.0012, material="steel", contype="0", conaffinity="0")}
        {_cyl_between((x + 0.04, yb, zc + 0.07), (x + 0.04, -W / 2, zc + 0.10), 0.0012, material="steel", contype="0", conaffinity="0")}
      </body>"""


def _bowl_xml():
    """Heavy ceramic crock built from tilted wall segments (true concave shape)."""
    cx, cy = BOWL_POS
    n, h, t = 30, 0.038, 0.007
    r_bot, r_top = BOWL_R - 0.012, BOWL_R
    g = [f'<geom name="bowl_base" type="cylinder" pos="0 0 {0.003}" size="{r_bot + 0.004} 0.003" material="ceramic"/>',
         f'<geom name="bowl_inner" type="cylinder" pos="0 0 {0.0062}" size="{r_bot - 0.002} 0.0002" material="ceramic_glaze" contype="0" conaffinity="0"/>']
    tilt = math.atan2(r_top - r_bot, h)
    seg = 2 * math.pi * (r_top + r_bot) / 2 / n
    for i in range(n):
        a = 2 * math.pi * i / n
        r = (r_top + r_bot) / 2 + t / 2
        g.append(f'<geom class="bowl_wall" pos="{r * math.cos(a):.5f} {r * math.sin(a):.5f} {h / 2:.4f}" '
                 f'euler="0 {tilt:.4f} {a:.4f}" size="{t / 2} {seg / 2 + 0.0012:.5f} {h / 2}"/>')
    rim = []
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        rr = r_top + t / 2
        rim.append(_cyl_between((rr * math.cos(a0), rr * math.sin(a0), h), (rr * math.cos(a1), rr * math.sin(a1), h),
                                t / 2 + 0.0008, material="ceramic", contype="0", conaffinity="0"))
    return (f'<body name="food_bowl" pos="{cx} {cy} {BED_Z}">\n        <freejoint name="bowl_free"/>\n        '
            + "\n        ".join(g + rim) + "\n      </body>")


def _pellets_xml(n=26, seed=3):
    """Extruded lab-chow blocks (~1.6 x 1.0 cm), each a free rigid body."""
    rng = np.random.default_rng(seed)
    cx, cy = BOWL_POS
    out = []
    k = 0
    while k < n:
        r = (BOWL_R - 0.022) * math.sqrt(rng.random())
        a = rng.uniform(0, 2 * math.pi)
        z = BED_Z + 0.014 + 0.012 * (k // 9)
        out.append(
            f'<body name="pellet{k}" pos="{cx + r * math.cos(a):.4f} {cy + r * math.sin(a):.4f} {z:.4f}" '
            f'euler="{rng.uniform(0, 3):.2f} {rng.uniform(0, 3):.2f} {rng.uniform(0, 3):.2f}">'
            f'<freejoint/><geom class="pellet" material="{"chow" if k % 3 else "chow_dark"}" '
            f'size="0.0048 {rng.uniform(0.0035, 0.0055):.4f}"/></body>')
        k += 1
    # a few pellets the rats have dragged out and scattered in the bedding
    for j in range(5):
        x = cx + rng.uniform(-0.15, 0.25)
        y = cy + rng.uniform(-0.08, 0.15)
        out.append(f'<body name="pellet_s{j}" pos="{x:.4f} {y:.4f} {BED_Z + 0.006}" euler="{math.pi / 2} 0 {rng.uniform(0, 3):.2f}">'
                   f'<freejoint/><geom class="pellet" material="chow" size="0.0048 0.0045"/></body>')
    return "\n      ".join(out)


def _house_xml():
    """Wooden hideout with an arched door on the front (-y) face."""
    hx, hy, hz = HOUSE_SIZE / 2
    t = 0.008
    door_w = 0.075
    side_w = (2 * hx - door_w) / 2
    g = [
        f'<geom class="wood" pos="0 {hy - t / 2} {hz}" size="{hx} {t / 2} {hz}"/>',
        f'<geom class="wood" pos="{-hx + t / 2} 0 {hz}" size="{t / 2} {hy} {hz}"/>',
        f'<geom class="wood" pos="{hx - t / 2} 0 {hz}" size="{t / 2} {hy} {hz}"/>',
        f'<geom class="wood" pos="{-hx + side_w / 2} {-hy + t / 2} {hz}" size="{side_w / 2} {t / 2} {hz}"/>',
        f'<geom class="wood" pos="{hx - side_w / 2} {-hy + t / 2} {hz}" size="{side_w / 2} {t / 2} {hz}"/>',
        f'<geom class="wood" pos="0 {-hy + t / 2} {2 * hz - 0.02}" size="{door_w / 2} {t / 2} 0.02"/>',
        f'<geom class="wood" pos="0 {-hy + t / 2} {2 * hz - 0.04}" size="{door_w / 2} {t / 2} 0.008" euler="1.5708 0 0" type="cylinder"/>',
        # pitched roof
        f'<geom class="wood_roof" pos="0 {-hy / 2 - 0.002} {2 * hz + 0.022}" euler="0.5 0 0" size="{hx + 0.012} {hy * 0.62} 0.005"/>',
        f'<geom class="wood_roof" pos="0 {hy / 2 + 0.002} {2 * hz + 0.022}" euler="-0.5 0 0" size="{hx + 0.012} {hy * 0.62} 0.005"/>',
    ]
    return (f'<body name="hideout" pos="{HOUSE_POS[0]} {HOUSE_POS[1]} {BED_Z}">\n        '
            + "\n        ".join(g) + "\n      </body>")


def _shelf_xml():
    """Wire half-shelf with a fleece liner and a wooden ramp, back-left corner."""
    x0, x1, y0, y1, z = -L / 2, -0.06, 0.02, W / 2, 0.33
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    g = [f'<geom class="frame" fromto="{x0} {y0} {z} {x1} {y0} {z}"/>',
         f'<geom class="frame" fromto="{x1} {y0} {z} {x1} {y1} {z}"/>',
         f'<geom name="shelf_liner" type="box" pos="{cx} {cy} {z + 0.003}" size="{(x1 - x0) / 2 - 0.005} {(y1 - y0) / 2 - 0.005} 0.003" material="fleece"/>']
    for x in np.arange(x0, x1, 0.025):
        g.append(f'<geom class="frame_thin" fromto="{x:.4f} {y0} {z} {x:.4f} {y1} {z}"/>')
    # ramp from the shelf down to the bedding (along +x, against the back bars)
    rz0, rz1 = BED_Z, z
    rx0, rx1 = x1 + 0.38, x1 - 0.005
    ry = y1 - 0.045
    ang = math.atan2(rz1 - rz0, rx0 - rx1)
    length = math.hypot(rz1 - rz0, rx0 - rx1)
    g.append(f'<geom name="ramp" class="wood" pos="{(rx0 + rx1) / 2:.4f} {ry} {(rz0 + rz1) / 2 + 0.004:.4f}" '
             f'euler="0 {ang:.4f} 0" size="{length / 2:.4f} 0.04 0.004"/>')
    for k in range(1, 12):
        s = k / 12
        g.append(f'<geom class="wood_slat" pos="{rx0 + (rx1 - rx0) * s:.4f} {ry} {rz0 + (rz1 - rz0) * s + 0.009:.4f}" '
                 f'euler="0 {ang:.4f} 0" size="0.003 0.038 0.0022"/>')
    return "\n      ".join(g)


def cage_xml():
    textures.ensure_textures()
    tex = os.path.join(ASSETS, "textures")
    return f"""
<mujoco model="rat_cage">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" iterations="30" integrator="implicitfast"/>
  <size memory="200M"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="40"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.40 0.40 0.40" diffuse="0.30 0.30 0.30" specular="0.05 0.05 0.05"/>
    <map znear="0.005" zfar="20" shadowclip="2.5" haze="0.0"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.85 0.85 0.82" rgb2="0.55 0.55 0.52" width="256" height="256"/>
    <texture name="floor_wood" type="2d" file="{tex}/floor_wood.png"/>
    <texture name="light_wood" type="2d" file="{tex}/light_wood.png"/>
    <texture name="bedding" type="2d" file="{tex}/bedding.png"/>
    <texture name="wall" type="2d" file="{tex}/wall.png"/>
    <material name="floor" texture="floor_wood" texrepeat="2 2" texuniform="true" reflectance="0.08" specular="0.3" shininess="0.5"/>
    <material name="wall" texture="wall" texrepeat="3 3" texuniform="true" specular="0.05"/>
    <material name="bedding" texture="bedding" texrepeat="5 5" texuniform="true" specular="0.0" shininess="0"/>
    <material name="flake" rgba="0.95 0.91 0.81 1" specular="0"/>
    <material name="flake_dark" rgba="0.74 0.62 0.45 1" specular="0"/>
    <material name="pan" rgba="0.10 0.14 0.26 1" specular="0.45" shininess="0.6" reflectance="0.04"/>
    <material name="wire" rgba="0.13 0.13 0.14 1" specular="0.8" shininess="0.9"/>
    <material name="steel" rgba="0.72 0.73 0.75 1" specular="1" shininess="1" reflectance="0.3"/>
    <material name="bottle_plastic" rgba="0.86 0.93 0.97 0.22" specular="0.9" shininess="1"/>
    <material name="water" rgba="0.45 0.66 0.90 0.38" specular="0.8" shininess="1"/>
    <material name="water_top" rgba="0.70 0.85 1.0 0.55" specular="1" shininess="1"/>
    <material name="water_drop" rgba="0.75 0.88 1.0 0.7" specular="1" shininess="1"/>
    <material name="label" rgba="0.15 0.55 0.35 0.9" specular="0.2"/>
    <material name="cap" rgba="0.15 0.55 0.35 1" specular="0.4"/>
    <material name="ceramic" rgba="0.60 0.32 0.20 1" specular="0.35" shininess="0.6"/>
    <material name="ceramic_glaze" rgba="0.93 0.90 0.82 1" specular="0.7" shininess="0.9" reflectance="0.05"/>
    <material name="chow" rgba="0.69 0.56 0.36 1" specular="0.05"/>
    <material name="chow_dark" rgba="0.58 0.46 0.29 1" specular="0.05"/>
    <material name="wood" texture="light_wood" texrepeat="4 4" texuniform="true" specular="0.15"/>
    <material name="fleece" rgba="0.42 0.22 0.45 1" specular="0"/>
    <material name="table" texture="floor_wood" texrepeat="3 3" texuniform="true" specular="0.3"/>
  </asset>

  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="wire"><geom type="capsule" size="{BAR_R}" material="wire" contype="0" conaffinity="0"/></default>
    <default class="frame"><geom type="capsule" size="0.003" material="wire" contype="0" conaffinity="0"/></default>
    <default class="frame_thin"><geom type="capsule" size="0.0018" material="wire" contype="0" conaffinity="0"/></default>
    <default class="post"><geom type="box" material="wire" contype="0" conaffinity="0"/></default>
    <default class="latch"><geom type="box" material="wire" contype="0" conaffinity="0"/></default>
    <default class="pan"><geom type="box" material="pan"/></default>
    <default class="pan_lip"><geom type="capsule" size="0.006" material="pan" contype="0" conaffinity="0"/></default>
    <default class="wallcol"><geom type="box" rgba="0 0 0 0" group="3"/></default>
    <default class="flake"><geom type="box" contype="0" conaffinity="0"/></default>
    <default class="bowl_wall"><geom type="box" material="ceramic" density="2400" friction="0.8 0.005 0.0001"/></default>
    <default class="pellet"><geom type="capsule" density="1300" friction="0.9 0.01 0.0001" condim="3" solref="0.01 1" solimp="0.9 0.95 0.001"/></default>
    <default class="wood"><geom type="box" material="wood" density="600"/></default>
    <default class="wood_roof"><geom type="box" material="wood" contype="0" conaffinity="0"/></default>
    <default class="wood_slat"><geom type="box" material="wood" contype="0" conaffinity="0"/></default>
  </default>

  <worldbody>
    <!-- room -->
    <geom name="room_floor" type="plane" size="4 4 0.1" pos="0 0 -0.72" material="floor"/>
    <geom name="back_wall" type="box" pos="0 1.1 0.6" size="3 0.02 1.6" material="wall" contype="0" conaffinity="0"/>
    <geom name="side_wall" type="box" pos="-1.6 0 0.6" size="0.02 3 1.6" material="wall" contype="0" conaffinity="0"/>
    <!-- sturdy table the cage stands on -->
    <geom name="table_top" type="box" pos="0 0.05 -0.02" size="0.62 0.45 0.02" material="table"/>
    <geom type="box" pos="-0.57 -0.35 -0.37" size="0.025 0.025 0.33" material="table" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.57 -0.35 -0.37" size="0.025 0.025 0.33" material="table" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.57 0.45 -0.37" size="0.025 0.025 0.33" material="table" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.57 0.45 -0.37" size="0.025 0.025 0.33" material="table" contype="0" conaffinity="0"/>

    <!-- lighting: soft ceiling key light + fill + window light -->
    <light name="key" pos="0.4 -0.6 2.2" dir="-0.15 0.25 -1" diffuse="0.40 0.39 0.37" specular="0.2 0.2 0.2" castshadow="false" cutoff="60" exponent="2"/>
    <light name="shadow" pos="0.1 -0.2 2.4" dir="0 0.05 -1" diffuse="0.16 0.16 0.15" specular="0 0 0" castshadow="true" cutoff="45"/>
    <light name="fill" pos="-1.2 -1.4 1.6" dir="0.6 0.7 -0.6" diffuse="0.25 0.26 0.29" specular="0.05 0.05 0.05" castshadow="false" cutoff="70"/>
    <light name="window" directional="true" pos="1.5 0.5 2" dir="-0.6 -0.2 -0.8" diffuse="0.16 0.16 0.15" castshadow="false"/>

    <!-- cameras -->
    {_cam("main", (0.30, -0.98, 0.74), (0.0, 0.02, 0.10), 42)}
    {_cam("front_low", (0.05, -0.92, 0.36), (0.0, 0.0, 0.08), 46)}
    {_cam("water_close", (0.74, -0.30, 0.27), (WATER_X - 0.03, -0.20, 0.08), 36)}
    {_cam("food_close", (-0.72, -0.34, 0.30), (BOWL_POS[0] + 0.06, BOWL_POS[1] - 0.03, 0.05), 38)}
    {_cam("top", (0.0, -0.02, 1.45), (0.0, 0.0, 0.0), 42)}
    {_cam("orbit", (0.0, -1.2, 0.8), (0.0, 0.0, 0.1), 40)}

    <!-- cage -->
    <body name="cage">
      {_pan_xml()}
      {_bars_xml()}
      {_shelf_xml()}
    </body>
    {_water_bottle_xml()}
    {_house_xml()}
    <body name="chew_block" pos="{CHEW_POS[0]} {CHEW_POS[1]} {BED_Z + 0.015}" euler="0 0 0.4">
      <geom class="wood" size="0.03 0.018 0.015"/>
      <geom type="capsule" fromto="0 0 0.015 0 0 0.05" size="0.002" material="steel" contype="0" conaffinity="0"/>
    </body>
    {_bowl_xml()}
    {_pellets_xml()}
  </worldbody>
</mujoco>
"""


RAT_STYLES = {
    "rat1": dict(texture="fur_white.png", label="albino"),
    "rat2": dict(texture="fur_agouti.png", label="agouti"),
}


def build_spec(rat_starts=None):
    """Returns an MjSpec of the cage with two rats attached."""
    skin_path = skin_uv.build()
    spec = mujoco.MjSpec.from_string(cage_xml())
    if rat_starts is None:
        rat_starts = {"rat1": ((-0.10, -0.01), 0.15), "rat2": ((0.10, -0.12), 2.2)}
    for name, ((x, y), yaw) in rat_starts.items():
        rat = mujoco.MjSpec.from_file(os.path.join(ASSETS, "rodent.xml"))
        # use UV-mapped skin + fur material
        rat.njmax = -1
        rat.nconmax = -1
        style = RAT_STYLES[name]
        tex = rat.add_texture(name="fur", type=mujoco.mjtTexture.mjTEXTURE_2D,
                              file=os.path.join(ASSETS, "textures", style["texture"]))
        mat = rat.add_material(name="fur", rgba=[1, 1, 1, 1], specular=0.12, shininess=0.15,
                               texrepeat=[1, 1])
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "fur"
        rat.skins[0].file = skin_path
        rat.skins[0].material = "fur"
        torso = rat.body("torso")
        for c in list(torso.cameras):
            rat.delete(c)
        for li in list(torso.lights):
            rat.delete(li)
        torso.add_freejoint(name="root")
        # loose paper bedding gives less grip than the default paw friction (1.5)
        for g in rat.geoms:
            if g.classname.name == "collision_primitive_paw":
                g.friction = [0.9, 0.005, 0.0001]
        frame = spec.worldbody.add_frame(pos=[x, y, BED_Z + 0.004],
                                         quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        frame.attach_body(torso, f"{name}/", "")
    return spec


def build_model(**kw):
    spec = build_spec(**kw)
    model = spec.compile()
    return spec, model


if __name__ == "__main__":
    spec, model = build_model()
    print(f"bodies={model.nbody} geoms={model.ngeom} dof={model.nv} actuators={model.nu} skins={model.nskin}")
    with open(os.path.join(HERE, "rat_cage_scene.xml"), "w") as f:
        f.write(spec.to_xml())
