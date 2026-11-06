"""Expansion experiments for Paper 2 (journal version):
  1. multi_calib  : value maps of a light and a full System 2, gains measured by an
                    independent reference planner (M=32) -> results/multi_calib.json
  2. multi_eval   : multi-planner gate vs single-planner gates at a compute budget
  3. dense_*      : second scenario (24 vehicles, 35% trucks): calibration + evaluation
  4. design6      : rollout count / horizon study with 6 realizations
  5. hmax         : sensitivity to the tolerable age
  6. mismatch     : System 2 imagining with a mis-specified process noise (Lemma 1)
Each stage is skipped if its result file exists."""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
from p2_core import *
from v2xsim import Config

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
Q = tuple(json.load(open(os.path.join(OUT, 'sysid.json'))).values())
BASE = dict(n_veh=16, look_slots=8)
DENSE = dict(n_veh=24, look_slots=8, truck_frac=0.35)
EP = dict(T_look=128, h_max_fac=2.0)
LITE = dict(M=4, H=16, c=0.125)          # cost ~ M*H relative to the full planner
FULL = dict(M=16, H=32, c=1.0)
RHO = 0.1; V = 10.0
G = {}


def exists(n):
    return os.path.exists(os.path.join(OUT, n + '.json'))


def save(n, o):
    json.dump(o, open(os.path.join(OUT, n + '.json'), 'w'))


def vm_from(knots, vals):
    m = ValueMap(); m.knots = np.asarray(knots); m.vals = np.asarray(vals); return m


def job(a):
    tag, ckw, kw, seed = a
    kw = dict(kw)
    if 'vmaps_ref' in kw:
        kw['vmaps'] = [vm_from(*G[r]) for r in kw.pop('vmaps_ref')]
    if 'vmap_ref' in kw:
        kw['vmap'] = vm_from(*G[kw.pop('vmap_ref')])
    params = dict(EP); params.update(kw)
    params.setdefault('n_slots', 6000)
    t = time.time()
    r = run_episode(Config(**ckw), seed=seed, q=Q, **params)
    r['wall'] = time.time() - t; r['seed'] = seed
    return tag, r


def run(jobs, seeds):
    allj = [(tag, ck, kw, s) for (tag, ck, kw) in jobs for s in seeds]
    with Pool(2) as P:
        res = P.map(job, allj, chunksize=1)
    out = {}
    for tag, r in res:
        out.setdefault(tag, []).append(r)
    return out


def cj(seed):
    return run_episode(Config(**BASE), gate='s1', seed=seed, n_slots=4000, q=Q, planners=[LITE, FULL],
                       log_multi=True, **EP)['mlog']


def cd(seed):
    return run_episode(Config(**DENSE), gate='s1', seed=seed, n_slots=4000, q=Q, log_calib=True, **EP)['calib']


def lam_for_budget(v, gmat, costs, rho):
    """Threshold lambda of the (multi-planner) Lagrangian gate meeting the budget
    on calibration data: choose argmax_j (g_j(v) - lambda c_j), with option 0 free."""
    lo, hi = 0.0, max(1e-6, gmat.max() / min(costs)) * 2
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        sc = np.concatenate([np.zeros((len(v), 1)), gmat - lam * np.asarray(costs)[None, :]], axis=1)
        j = np.argmax(sc, axis=1)
        use = np.where(j > 0, np.asarray([0.0] + list(costs))[j], 0.0).mean()
        if use > rho:
            lo = lam
        else:
            hi = lam
    return hi


if __name__ == '__main__':
    t0 = time.time()
    # ---------- 1. multi-planner calibration ----------
    if not exists('multi_calib'):
        with Pool(2) as P:
            C = P.map(cj, [100, 101])
        v = np.concatenate([np.array(c['v']) for c in C]); g = np.concatenate([np.array(c['g']) for c in C])
        maps = [ValueMap().fit(v, g[:, i]) for i in range(2)]
        gm = np.stack([np.interp(v, maps[i].knots, maps[i].vals) for i in range(2)], 1)
        lam = dict(both=lam_for_budget(v, gm, [LITE['c'], FULL['c']], RHO),
                   full=lam_for_budget(v, gm[:, 1:2], [FULL['c']], RHO),
                   lite=lam_for_budget(v, gm[:, 0:1], [LITE['c']], RHO))
        save('multi_calib', dict(v=v.tolist(), g=g.tolist(), knots=[m.knots.tolist() for m in maps],
                                 vals=[m.vals.tolist() for m in maps], lam=lam))
        print('multi_calib', time.time() - t0, flush=True)
    mc = json.load(open(os.path.join(OUT, 'multi_calib.json')))
    G['lite'] = (mc['knots'][0], mc['vals'][0]); G['full'] = (mc['knots'][1], mc['vals'][1])

    # ---------- 2. multi-planner evaluation ----------
    if not exists('multi_eval'):
        lam = mc['lam']
        jobs = [('s1', BASE, dict(gate='s1')),
                ('s2_full', BASE, dict(gate='s2', M=16, H=32)),
                ('s2_lite', BASE, dict(gate='s2', M=4, H=16)),
                ('vq_full', BASE, dict(gate='multi', rho=RHO, V=V, planners=[FULL], vmaps_ref=['full'], Q0=V * lam['full'])),
                ('vq_lite', BASE, dict(gate='multi', rho=RHO, V=V, planners=[LITE], vmaps_ref=['lite'], Q0=V * lam['lite'])),
                ('vq_multi', BASE, dict(gate='multi', rho=RHO, V=V, planners=[LITE, FULL], vmaps_ref=['lite', 'full'],
                                        Q0=V * lam['both']))]
        save('multi_eval', run(jobs, list(range(8))))
        print('multi_eval', time.time() - t0, flush=True)

    # ---------- 3. dense scenario ----------
    if not exists('dense_calib'):
        with Pool(2) as P:
            C = P.map(cd, [100, 101])
        v = np.concatenate([np.array(c['v']) for c in C]); g = np.concatenate([np.array(c['g']) for c in C])
        m = ValueMap().fit(v, g)
        lam = lam_for_budget(v, np.interp(v, m.knots, m.vals)[:, None], [1.0], RHO)
        save('dense_calib', dict(knots=m.knots.tolist(), vals=m.vals.tolist(), thr=float(np.quantile(v, 1 - RHO)), lam=lam,
                                 v=v.tolist(), g=g.tolist()))
        print('dense_calib', time.time() - t0, flush=True)
    dc = json.load(open(os.path.join(OUT, 'dense_calib.json')))
    G['dense'] = (dc['knots'], dc['vals'])
    if not exists('dense_eval'):
        jobs = [('s1', DENSE, dict(gate='s1')), ('s2', DENSE, dict(gate='s2')),
                ('random', DENSE, dict(gate='random', rho=RHO)),
                ('static', DENSE, dict(gate='static', rho=RHO, thr=dc['thr'])),
                ('vq', DENSE, dict(gate='vq', rho=RHO, V=V, vmap_ref='dense', Q0=V * dc['lam']))]
        save('dense_eval', run(jobs, list(range(6))))
        print('dense_eval', time.time() - t0, flush=True)

    # ---------- 4. design study, 6 realizations ----------
    if not exists('design6'):
        jobs = [('S1', BASE, dict(gate='s1', n_slots=3000))]
        jobs += [(f'M{M}', BASE, dict(gate='s2', M=M, n_slots=3000)) for M in [2, 4, 8, 16]]
        jobs += [(f'H{H}', BASE, dict(gate='s2', H=H, n_slots=3000)) for H in [8, 16, 64]]
        save('design6', run(jobs, list(range(6))))
        print('design6', time.time() - t0, flush=True)

    # ---------- 5. tolerable-age sensitivity ----------
    cal = json.load(open(os.path.join(OUT, 'calibration.json')))
    G['main'] = (cal['knots'], cal['vals'])
    lam_main = lam_for_budget(np.array(cal['cal']['v']),
                              np.interp(cal['cal']['v'], cal['knots'], cal['vals'])[:, None], [1.0], RHO)
    if not exists('hmax'):
        jobs = []
        for hf in [1.5, 3.0]:
            jobs += [(f's1_{hf}', BASE, dict(gate='s1', h_max_fac=hf)),
                     (f'static_{hf}', BASE, dict(gate='static', rho=RHO, thr=cal['thr']['0.1']['static'], h_max_fac=hf)),
                     (f'vq_{hf}', BASE, dict(gate='vq', rho=RHO, V=V, vmap_ref='main', Q0=V * lam_main, h_max_fac=hf))]
        save('hmax', run(jobs, list(range(4))))
        print('hmax', time.time() - t0, flush=True)

    # ---------- 6. world-model mismatch in System 2 ----------
    if not exists('mismatch'):
        jobs = [(f'q{sc}', BASE, dict(gate='s2', s2_qscale=sc, n_slots=3000)) for sc in [0.01, 100.0]]
        save('mismatch', run(jobs, list(range(6))))
        print('mismatch', time.time() - t0, flush=True)
    print('all done', time.time() - t0, flush=True)
