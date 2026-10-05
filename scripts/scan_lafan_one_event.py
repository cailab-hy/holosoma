"""Scan all retargeted LAFAN clips (G1, original 30 fps) for windows with exactly one high-risk event
(single-support high leg lift, or a flight phase), calm standing at both ends, upright torso, small travel."""
import glob, json, os, sys
import numpy as np, mujoco
FPS = 30
SP = sys.argv[1]
CALM_JV, CALM_PV, CALM_D, CALM_TILT, CALM_FOOT, PATH, TILT, WMIN, WMAX = map(float, sys.argv[2:11])
OUT = sys.argv[11]
END_MODE = sys.argv[12] if len(sys.argv) > 12 else 'calm'   # 'calm' (append) or 'upright' (no append, like dance1)
m = mujoco.MjModel.from_xml_path("models/g1/g1_29dof.xml"); d = mujoco.MjData(m)
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
L, R, T, P = (bn.index(x) for x in ("left_ankle_roll_link", "right_ankle_roll_link", "torso_link", "pelvis"))
dflt = json.load(open(f"{SP}/g1_default_angles.json")); d0 = np.array([dflt[n] for n in jn])
def frames(q):
    lf = np.empty(len(q)); rf = np.empty(len(q)); tilt = np.empty(len(q)); pz = np.empty(len(q))
    for i, qq in enumerate(q):
        d.qpos[:] = qq; mujoco.mj_kinematics(m, d)
        lf[i], rf[i], pz[i] = d.xpos[L, 2], d.xpos[R, 2], d.xpos[P, 2]
        tilt[i] = np.degrees(np.arccos(np.clip(d.xmat[T].reshape(3, 3)[2, 2], -1, 1)))
    g = np.percentile(np.minimum(lf, rf), 1)
    return lf - g, rf - g, tilt, pz - g
def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out
rows = []
for f in sorted(glob.glob("demo_results_parallel/g1/robot_only/lafan/*_original.npz")):
    clip = os.path.basename(f).replace("_original.npz", "")
    q = np.load(f)["qpos"]; lf, rf, tilt, pz = frames(q)
    jq = q[:, 7:]; jv = np.abs(np.gradient(jq, axis=0)) * FPS
    sm = lambda x, k=9: np.convolve(x, np.ones(k) / k, mode="same")
    pv = sm(np.linalg.norm(np.gradient(q[:, :2], axis=0), axis=1) * FPS); jvm = sm(jv.mean(1))
    dist = np.linalg.norm(jq - d0, axis=1)
    hi, lo = np.maximum(lf, rf), np.minimum(lf, rf)
    event = ((hi > 0.30) & (lo < 0.06)) | (lo > 0.06)          # high single-leg lift or flight
    stepish = (hi > 0.10)                                        # any foot off the ground
    calm = (lf < CALM_FOOT) & (rf < CALM_FOOT) & (pv < CALM_PV) & (jvm < CALM_JV) & (dist < CALM_D) & (tilt < CALM_TILT)
    ev = [(s, e) for s, e in runs(event) if e - s >= 3]
    cl = []                                                      # merge events < 1 s apart
    for s, e in ev:
        if cl and s - cl[-1][1] < FPS: cl[-1] = (cl[-1][0], e)
        else: cl.append((s, e))
    calm_ok = np.array([calm[max(0, i - 4):i + 5].all() for i in range(len(q))])  # calm for 0.3 s around frame
    upright = (lf < CALM_FOOT) & (rf < CALM_FOOT) & (tilt < CALM_TILT)
    end_ok = calm_ok if END_MODE == 'calm' else np.array([upright[max(0, i - 4):i + 5].all() for i in range(len(q))])
    for (cs, ce) in cl:
        best = None
        for s in range(max(0, ce - int(WMAX * FPS)), cs - FPS + 1, 3):          # event >= 1 s after start
            if not calm_ok[s]: continue
            for e in range(max(ce + FPS, s + int(WMIN * FPS)), min(len(q) - 1, s + int(WMAX * FPS)) + 1, 3):  # >= 1 s after event, 5-7 s long
                if not end_ok[e]: continue
                inside = [c for c in cl if c[1] > s and c[0] < e]
                if len(inside) != 1: continue
                travel = np.linalg.norm(q[e, :2] - q[s, :2]); path_max = np.linalg.norm(q[s:e, :2] - q[s, :2], axis=1).max()
                if path_max > PATH: continue
                maxtilt = tilt[s:e].max()
                if maxtilt > TILT: continue
                nsteps = len([r for r in runs(stepish[s:e]) if r[1] - r[0] >= 3])
                pos = (cs - s) / (e - s)
                if not (0.15 <= pos <= 0.75): continue
                score = -(e - s) / FPS + 0.3 * abs(pos - 0.4)   # longest window first, event not at the edges
                if best is None or score < best[0]:
                    best = (score, s, e, travel, path_max, maxtilt, nsteps, pos)
        if best:
            score, s, e, travel, path_max, maxtilt, nsteps, pos = best
            kind = "flight" if (lo[cs:ce] > 0.06).any() else ("L" if lf[cs:ce].max() > rf[cs:ce].max() else "R") + "-leg lift"
            rows.append(dict(clip=clip, s=s, e=e, dur=(e - s) / FPS, ev_t=(cs - s) / FPS, ev_dur=(ce - cs) / FPS, kind=kind,
                             peak_foot=float(hi[cs:ce].max()), max_tilt=float(maxtilt), path=float(path_max), steps=nsteps,
                             jv_max=float(jv[s:e].max()), jv_p99=float(np.percentile(jv[s:e], 99)), d_start=float(dist[s]), d_end=float(dist[e]),
                             min_pelvis=float(pz[s:e].min()), score=float(score)))
rows.sort(key=lambda r: r["score"])
json.dump(rows, open(f"{SP}/{OUT}", "w"), indent=1)
print(f"{len(rows)} candidate windows")
print(f"{'clip':28s} {'frames':>11s} {'dur':>4s} {'event':>13s} {'@s':>4s} {'len':>4s} {'foot':>5s} {'tilt':>5s} {'path':>5s} {'steps':>5s} {'jv p99/max':>11s} {'d0 s/e':>9s} {'pelv':>5s}")
for r in rows:
    print(f"{r['clip']:28s} {r['s']:5d}-{r['e']:<5d} {r['dur']:4.1f} {r['kind']:>13s} {r['ev_t']:4.1f} {r['ev_dur']:4.1f} {r['peak_foot']:5.2f} {r['max_tilt']:5.0f} {r['path']:5.2f} {r['steps']:5d} {r['jv_p99']:5.1f}/{r['jv_max']:4.1f} {r['d_start']:4.2f}/{r['d_end']:4.2f} {r['min_pelvis']:5.2f}")
