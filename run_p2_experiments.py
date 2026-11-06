"""Regenerates every number and figure of Paper 2 (results/*.json).
Usage: python run_p2_experiments.py [--quick]"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
from p2_core import *
from v2xsim import learn_process_noise

QUICK = '--quick' in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'); os.makedirs(OUT, exist_ok=True)
NS = 2000 if QUICK else 6000
NCAL = 1500 if QUICK else 4000
SEEDS = [0, 1] if QUICK else [0, 1, 2]
CFG = dict(n_veh=16, look_slots=8)
EP = dict(T_look=128, h_max_fac=2.0)
RHOS = [0.05, 0.1, 0.2, 0.3]


def save(name, obj):
    with open(os.path.join(OUT, name + '.json'), 'w') as f:
        json.dump(obj, f)


def job(a):
    tag, kw, seed = a
    t = time.time()
    r = run_episode(Config(**CFG), seed=seed, n_slots=kw.pop('n_slots', NS), q=Q, vmap=VMAP, **EP, **kw)
    r['wall'] = time.time() - t
    return tag, seed, r


def sweep(jobs, name, seeds=None):
    seeds = SEEDS if seeds is None else seeds
    allj = [(tag, dict(kw), s) for (tag, kw) in jobs for s in seeds]
    with Pool(2) as P:
        res = P.map(job, allj, chunksize=1)
    out = {}
    for tag, s, r in res:
        out.setdefault(tag, []).append(r)
    save(name, out)
    return out


def calib_job(seed):
    return run_episode(Config(**CFG), gate='s1', seed=seed, n_slots=NCAL, q=Q, log_calib=True, **EP)['calib']


if __name__ == '__main__':
    t0 = time.time()
    qx, qy = learn_process_noise(Config(**CFG), seed=999, T_obs_s=30.0, every=50)
    Q = (qx, qy); VMAP = None
    save('sysid', dict(q_x=qx, q_y=qy))
    # --- 1. Calibration phase (seeds disjoint from evaluation) ---
    with Pool(2) as P:
        C = P.map(calib_job, [100, 101])
    cal = {k: np.concatenate([np.asarray(c[k]) for c in C]).tolist() for k in C[0]}
    v = np.array(cal['v']); g = np.array(cal['g']); unc = np.array(cal['unc'])
    VMAP = ValueMap().fit(v, g)
    thr = {}
    for rho in RHOS:
        thr[rho] = dict(static=float(np.quantile(v, 1 - rho)), unc=float(np.quantile(unc, 1 - rho)),
                        oracle=float(np.quantile(g, 1 - rho)))
    save('calibration', dict(cal=cal, knots=VMAP.knots.tolist(), vals=VMAP.vals.tolist(),
                             thr={str(k): w for k, w in thr.items()}))
    print('calibration done', time.time() - t0, flush=True)

    # --- 2. Main comparison over the deliberation budget rho ---
    jobs = [('s1', dict(gate='s1')), ('s2', dict(gate='s2'))]
    for rho in RHOS:
        jobs += [(f'random_{rho}', dict(gate='random', rho=rho)),
                 (f'periodic_{rho}', dict(gate='periodic', rho=rho)),
                 (f'unc_{rho}', dict(gate='unc', rho=rho, unc_thr=thr[rho]['unc'])),
                 (f'static_{rho}', dict(gate='static', rho=rho, thr=thr[rho]['static'])),
                 (f'vq_{rho}', dict(gate='vq', rho=rho, V=50.0, Q0=50.0 * VMAP(thr[rho]['static'])))]
    sweep(jobs, 'main_rho')
    print('main done', time.time() - t0, flush=True)

    # --- 3. Virtual-queue trade-off in V (Theorem 5) with budget traces ---
    jobs = [(f'vq_V{V}', dict(gate='vq', rho=0.1, V=V, Q0=V * VMAP(thr[0.1]['static']), record_trace=True))
            for V in [0.5, 2.0, 10.0, 50.0, 250.0]]
    jobs += [('vq_V50.0_cold', dict(gate='vq', rho=0.1, V=50.0, Q0=0.0, record_trace=True))]
    sweep(jobs, 'V_sweep')
    print('V done', time.time() - t0, flush=True)

    # --- 4. Oracle gate (needs System 2 every slot) ---
    sweep([('oracle_0.1', dict(gate='oracle', rho=0.1, thr=thr[0.1]['oracle']))], 'oracle', seeds=SEEDS[:2])
    print('oracle done', time.time() - t0, flush=True)

    # --- 5. System 2 design: rollout count M and horizon H (Theorems 2-3) ---
    jobs = [('S1', dict(gate='s1', n_slots=NS // 2))]
    jobs += [(f'M{M}', dict(gate='s2', M=M, n_slots=NS // 2)) for M in [2, 4, 8, 16, 32]]
    jobs += [(f'H{H}', dict(gate='s2', H=H, n_slots=NS // 2)) for H in [4, 8, 16, 64]]
    sweep(jobs, 'design', seeds=SEEDS[:2])
    print('all done', time.time() - t0, flush=True)
