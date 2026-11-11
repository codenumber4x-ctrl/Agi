"""Adds cylindrical UV texture coordinates to the virtual-rodent skin.

The original ``rodent_walker_skin.skn`` (from dm_control) has no texcoords,
so a fur texture cannot be mapped onto it. MuJoCo's .skn layout is:
  int32[4] header (nvert, ntexcoord, nface, nbone)
  float32[3*nvert] vertices, float32[2*ntexcoord] texcoords, int32[3*nface] faces
  then per bone: char[40] name, float32[3] bindpos, float32[4] bindquat,
                 int32 vcount, int32[vcount] ids, float32[vcount] weights
We insert texcoords computed by a cylindrical projection around the body's
long (x) axis and copy everything else unchanged.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "assets", "rodent_walker_skin.skn")
DST = os.path.join(HERE, "assets", "rodent_skin_uv.skn")


def build(force=False):
    if os.path.exists(DST) and not force:
        return DST
    raw = open(SRC, "rb").read()
    nvert, ntex, nface, nbone = np.frombuffer(raw[:16], np.int32)
    assert ntex == 0
    off = 16
    verts = np.frombuffer(raw[off:off + 12 * nvert], np.float32).reshape(-1, 3)
    off += 12 * nvert
    rest = raw[off:]
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2] - 0.04
    u = (x - x.min()) / 0.06                     # texture repeats every 6 cm
    v = (np.arctan2(z, y) + np.pi) / (2 * np.pi) * 3.0
    uv = np.stack([u, v], 1).astype(np.float32)
    header = np.array([nvert, nvert, nface, nbone], np.int32).tobytes()
    with open(DST, "wb") as f:
        f.write(header + verts.astype(np.float32).tobytes() + uv.tobytes() + rest)
    return DST


if __name__ == "__main__":
    print(build(force=True))
