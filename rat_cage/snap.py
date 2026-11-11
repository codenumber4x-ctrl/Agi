"""Quick still renders of the scene from every camera (no simulation)."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, imageio.v2 as imageio
import cage

def render_all(model, data, prefix, w=1280, h=720, cams=None):
    r = mujoco.Renderer(model, h, w)
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = opt.geomgroup[3] = 0   # hide collision proxies
    out = []
    for cam in cams or [model.camera(i).name for i in range(model.ncam) if "/" not in model.camera(i).name]:
        r.update_scene(data, camera=cam, scene_option=opt)
        p = f"{prefix}_{cam}.png"
        imageio.imwrite(p, r.render()); out.append(p)
    r.close()
    return out

if __name__ == "__main__":
    spec, m = cage.build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(render_all(m, d, sys.argv[1] if len(sys.argv) > 1 else "screenshots/scene"))
