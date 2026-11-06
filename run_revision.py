"""Revision experiments (all results of the revised manuscript).

Changes with respect to the first submission
  * violation-weighted cost: psi = PSI_FAC * h_max (System 2 objective, gains and
    reported cost all use it), so the cost covered by the theory is the metric
    whose reduction is reported;
  * value maps fitted on gains measured by an independent reference planner
    (M_ref = 32), also for the single-planner gates (no winner's curse);
  * more traffic realizations, paired tests, two new budgeted baselines:
      'aq' online quantile tracking of the raw signal (fixed step),
      'pd' Lagrangian gate with diminishing dual step (CMDP-style primal-dual);
  * per-decision wall-clock timing of System 1 and System 2;
  * world-model perturbations of the imagined blockage and shadowing.
Each stage writes results_rev/<stage>.json and is skipped if the file exists.
Usage: python run_revision.py [stage ...]"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
from p2_core import *
from v2xsim import Config

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = os.environ.get('QUICK') == '1'
OUT = os.path.join(HERE, 'results_quick' if QUICK else 'results_rev'); os.makedirs(OUT, exist_ok=True)
Q = tuple(json.load(open(os.path.join(HERE, 'results', 'sysid.json'))).values())
BASE = dict(n_veh=16, look_slots=8)
DENSE = dict(n_veh=24, look_slots=8, truck_frac=0.35)
PSI_FAC = 10.0                       # psi = PSI_FAC * h_max per vehicle-slot above the limit
HF = 2.0                             # h_max = HF * K
LITE = dict(M=4, H=16, c=0.125)
FULL = dict(M=16, H=32, c=1.0)
V = 10.0; RHO = 0.1; RHOS = [0.05, 0.1, 0.2, 0.3]
N_MAIN = 20; N_FRONT = 6; N_MULTI = 10; N_DENSE = 6; N_DESIGN = 6; N_HMAX = 6; N_V = 3
NS = 6000; NCAL = 4000
if QUICK:
    N_MAIN = N_FRONT = N_MULTI = N_DENSE = N_DESIGN = N_HMAX = N_V = 2; NS = 150; NCAL = 120
ETA_AQ = 0.01; ETA0_PD = 1.0; V_MULTI = 10.0
MAPS = {}
TUNE_SEEDS = [100, 101]          # the calibration realizations (never used for evaluation)
TUNE_GRID = dict(vq=[0.5, 1.0, 2.0, 5.0, 10.0], aq=[1e-4, 1e-3, 1e-2], pd=[1.0, 10.0, 100.0],
                 multi=[0.125, 0.25, 0.5, 1.0, 2.0])


def apply_tuning():
    """Step sizes chosen on the calibration realizations: for every budget-enforcing
    gate, the setting with the lowest mean cost among those whose realized rate
    (compute) stays within 5% of the budget; if none does, the one closest to it."""
    global V, ETA_AQ, ETA0_PD, V_MULTI
    if os.path.exists(path('tune')):
        t = load('tune')['chosen']
        V, ETA_AQ, ETA0_PD, V_MULTI = t['vq'], t['aq'], t['pd'], t['multi']


def ep(cfg, hf=HF, **kw):
    K = cfg['n_veh']
    d = dict(T_look=128, h_max_fac=hf, psi=PSI_FAC * hf * K, n_slots=NS)
    d.update(kw)
    return d


def path(n):
    return os.path.join(OUT, n + '.json')


def save(n, o):
    json.dump(o, open(path(n), 'w'))


def load(n):
    return json.load(open(path(n)))


def vm(knots, vals, clip0=True):
    m = ValueMap(); m.knots = np.asarray(knots, float)
    m.vals = np.maximum(np.asarray(vals, float), 0.0) if clip0 else np.asarray(vals, float)
    return m


def job(a):
    tag, ckw, kw, seed = a
    kw = dict(kw)
    if 'vmap_ref' in kw:
        kw['vmap'] = vm(*MAPS[kw.pop('vmap_ref')])
    if 'vmaps_ref' in kw:
        kw['vmaps'] = [vm(*MAPS[r], clip0=False) for r in kw.pop('vmaps_ref')]
    t = time.time()
    r = run_episode(Config(**ckw), seed=seed, q=Q, **kw)
    r['wall'] = time.time() - t; r['seed'] = seed
    r.pop('trace', None) if not kw.get('record_trace') else None
    return tag, seed, r


def run(jobs, seeds, name=None):
    allj = [(tag, ck, kw, s) for (tag, ck, kw) in jobs for s in seeds]
    cost = lambda a: (a[2].get('n_slots', NS) * (1.0 if a[2].get('gate') in ('s2',) else 0.25
                      if a[2].get('rho', 0) >= 0.2 else 0.15))
    allj.sort(key=cost, reverse=True)
    out = {}
    with Pool(2) as P:
        for tag, s, r in P.imap_unordered(job, allj, chunksize=1):
            out.setdefault(tag, []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r['seed'])
    if name:
        save(name, out)
    return out


def lam_for_budget(v, gmat, costs, rho):
    lo, hi = 0.0, max(1e-6, gmat.max() / min(costs)) * 2
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        sc = np.concatenate([np.zeros((len(v), 1)), gmat - lam * np.asarray(costs)[None, :]], axis=1)
        j = np.argmax(sc, axis=1)
        use = np.where(j > 0, np.asarray([0.0] + list(costs))[j], 0.0).mean()
        lo, hi = (lam, hi) if use > rho else (lo, lam)
    return hi


def calib_base(seed):
    return run_episode(Config(**BASE), gate='s1', seed=seed, q=Q, planners=[LITE, FULL], log_multi=True,
                       **ep(BASE, n_slots=NCAL))['mlog']


def calib_dense(seed):
    return run_episode(Config(**DENSE), gate='s1', seed=seed, q=Q, planners=[FULL], log_multi=True,
                       **ep(DENSE, n_slots=NCAL))['mlog']


def fit_calibration(C, planners):
    v = np.concatenate([np.array(c['v']) for c in C]); g = np.concatenate([np.array(c['g']) for c in C])
    gs = np.concatenate([np.array(c['gself']) for c in C]); unc = np.concatenate([np.array(c['unc']) for c in C])
    maps = [ValueMap().fit(v, g[:, i]) for i in range(len(planners))]
    full = len(planners) - 1
    gfull = np.maximum(np.interp(v, maps[full].knots, maps[full].vals), 0.0)[:, None]
    thr = {str(r): dict(static=float(np.quantile(v, 1 - r)), unc=float(np.quantile(unc, 1 - r)),
                        lam=lam_for_budget(v, gfull, [1.0], r)) for r in RHOS}
    out = dict(v=v.tolist(), g=g.tolist(), gself=gs.tolist(), unc=unc.tolist(),
               knots=[m.knots.tolist() for m in maps], vals=[m.vals.tolist() for m in maps], thr=thr)
    if len(planners) == 2:
        gm = np.stack([np.interp(v, maps[i].knots, maps[i].vals) for i in range(2)], 1)
        out['lam_multi'] = dict(both=lam_for_budget(v, gm, [LITE['c'], FULL['c']], RHO),
                                full=lam_for_budget(v, gm[:, 1:2], [FULL['c']], RHO),
                                lite=lam_for_budget(v, gm[:, 0:1], [LITE['c']], RHO))
    return out


def gates(rho, cal, mapname, cfg, hf=HF, which=('random', 'periodic', 'unc', 'static', 'vq', 'aq', 'pd')):
    t = cal['thr'][str(rho)]
    d = dict(random=dict(gate='random', rho=rho), periodic=dict(gate='periodic', rho=rho),
             unc=dict(gate='unc', rho=rho, unc_thr=t['unc']), static=dict(gate='static', rho=rho, thr=t['static']),
             vq=dict(gate='vq', rho=rho, V=V, vmap_ref=mapname, Q0=V * t['lam']),
             aq=dict(gate='aq', rho=rho, thr=t['static'], eta=ETA_AQ),
             pd=dict(gate='pd', rho=rho, vmap_ref=mapname, lam0=t['lam'], eta0=ETA0_PD))
    return [(f'{w}_{rho}', cfg, ep(cfg, hf=hf, **d[w])) for w in which]


STAGES = ['calib', 'tune', 'main', 'main2', 'multi', 'frontier', 'hmax', 'vsweep', 'design', 'mismatch', 'dense_calib', 'dense', 'rho02']

if __name__ == '__main__':
    want = sys.argv[1:] or STAGES
    t0 = time.time()
    for stage in STAGES:
        if stage != 'calib' and (stage not in want or os.path.exists(path(stage))):
            continue
        if stage == 'calib':
            if not os.path.exists(path('calib')):
                with Pool(2) as P:
                    C = P.map(calib_base, [100, 101])
                save('calib', fit_calibration(C, [LITE, FULL]))
        cal = load('calib')
        apply_tuning()
        MAPS['lite'] = (cal['knots'][0], cal['vals'][0]); MAPS['full'] = (cal['knots'][1], cal['vals'][1])
        S = list(range(N_MAIN))
        if stage == 'tune':
            t = cal['thr'][str(RHO)]; lm = cal['lam_multi']; jobs = []
            for x in TUNE_GRID['vq']:
                jobs.append((f'vq|{x}', BASE, ep(BASE, gate='vq', rho=RHO, V=x, vmap_ref='full', Q0=x * t['lam'])))
            for x in TUNE_GRID['aq']:
                jobs.append((f'aq|{x}', BASE, ep(BASE, gate='aq', rho=RHO, thr=t['static'], eta=x)))
            for x in TUNE_GRID['pd']:
                jobs.append((f'pd|{x}', BASE, ep(BASE, gate='pd', rho=RHO, vmap_ref='full', lam0=t['lam'], eta0=x)))
            for x in TUNE_GRID['multi']:
                jobs.append((f'multi|{x}', BASE, ep(BASE, gate='multi', rho=RHO, V=x, planners=[LITE, FULL],
                                                    vmaps_ref=['lite', 'full'], Q0=x * lm['both'])))
            res = run(jobs, TUNE_SEEDS)
            chosen, table = {}, {}
            for g_ in TUNE_GRID:
                cand = []
                for x in TUNE_GRID[g_]:
                    rr = res[f'{g_}|{x}']
                    rate = float(np.mean([r['compute_rate'] for r in rr])); cost = float(np.mean([r['cost'] for r in rr]))
                    cand.append((x, rate, cost)); table[f'{g_}|{x}'] = dict(rate=rate, cost=cost)
                ok = [c for c in cand if c[1] <= 1.05 * RHO]
                chosen[g_] = min(ok, key=lambda c: c[2])[0] if ok else min(cand, key=lambda c: abs(c[1] - RHO))[0]
            save('tune', dict(chosen=chosen, table=table))
            print('tuned', chosen, flush=True)
            continue
        if stage == 'main2':
            # re-evaluate the three budget-enforcing gates with the tuned step sizes
            d = load('main')
            for k in ['vq_0.1', 'aq_0.1', 'pd_0.1']:
                if k in d and not any(kk.startswith(k.split('_')[0] + '_untuned') for kk in d):
                    d[k.split('_')[0] + '_untuned_0.1'] = d.pop(k)
            new = run(gates(RHO, cal, 'full', BASE, which=('vq', 'aq', 'pd')), S)
            d.update(new); save('main', d); save('main2', dict(V=V, eta=ETA_AQ, eta0=ETA0_PD))
            print(stage, 'done', round(time.time() - t0), flush=True)
            continue
        if stage == 'main':
            jobs = [('s1', BASE, ep(BASE, gate='s1', record_timing=True)),
                    ('s2', BASE, ep(BASE, gate='s2', record_timing=True))] + gates(RHO, cal, 'full', BASE)
            run(jobs, S, 'main')
        elif stage == 'multi':
            lm = cal['lam_multi']
            jobs = [('s2_lite', BASE, ep(BASE, gate='s2', M=4, H=16)),
                    ('vq_full', BASE, ep(BASE, gate='multi', rho=RHO, V=V_MULTI, planners=[FULL], vmaps_ref=['full'], Q0=V_MULTI * lm['full'])),
                    ('vq_lite', BASE, ep(BASE, gate='multi', rho=RHO, V=V_MULTI, planners=[LITE], vmaps_ref=['lite'], Q0=V_MULTI * lm['lite'])),
                    ('vq_multi', BASE, ep(BASE, gate='multi', rho=RHO, V=V_MULTI, planners=[LITE, FULL], vmaps_ref=['lite', 'full'],
                                          Q0=V_MULTI * lm['both']))]
            run(jobs, list(range(N_MULTI)), 'multi')
        elif stage == 'frontier':
            jobs = []
            for r in [0.05, 0.2, 0.3]:
                jobs += gates(r, cal, 'full', BASE, which=('random', 'static', 'vq', 'aq', 'pd'))
            run(jobs, list(range(N_FRONT)), 'frontier')
        elif stage == 'hmax':
            jobs = []
            for hf in [1.5, 3.0]:
                jobs += [(f's1_{hf}', BASE, ep(BASE, hf=hf, gate='s1'))]
                jobs += [(k.replace('_0.1', f'_{hf}'), c, kw) for k, c, kw in gates(RHO, cal, 'full', BASE, hf=hf, which=('static', 'vq'))]
            run(jobs, list(range(N_HMAX)), 'hmax')
        elif stage == 'vsweep':
            lam = cal['thr'][str(RHO)]['lam']
            jobs = [(f'vq_V{v_}', BASE, ep(BASE, gate='vq', rho=RHO, V=v_, vmap_ref='full', Q0=v_ * lam, record_trace=True))
                    for v_ in [0.25, 1.0, 4.0, 10.0, 50.0]]
            jobs += [('vq_V10.0_cold', BASE, ep(BASE, gate='vq', rho=RHO, V=10.0, vmap_ref='full', Q0=0.0, record_trace=True))]
            run(jobs, list(range(N_V)), 'vsweep')
        elif stage == 'design':
            jobs = [('S1', BASE, ep(BASE, gate='s1', n_slots=NS // 2))]
            jobs += [(f'M{m}', BASE, ep(BASE, gate='s2', M=m, n_slots=NS // 2)) for m in [2, 4, 8, 16]]
            jobs += [(f'H{h_}', BASE, ep(BASE, gate='s2', H=h_, n_slots=NS // 2)) for h_ in [8, 16, 64]]
            run(jobs, list(range(N_DESIGN)), 'design')
        elif stage == 'mismatch':
            jobs = [('q0.01', BASE, ep(BASE, gate='s2', s2_qscale=0.01, n_slots=NS // 2)),
                    ('q100', BASE, ep(BASE, gate='s2', s2_qscale=100.0, n_slots=NS // 2)),
                    ('noblock', BASE, ep(BASE, gate='s2', s2_block=0.0, n_slots=NS // 2)),
                    ('nosf', BASE, ep(BASE, gate='s2', s2_sf=0.0, n_slots=NS // 2))]
            run(jobs, list(range(N_DESIGN)), 'mismatch')
        elif stage == 'rho02':
            # rho = 0.2 on the remaining evaluation realizations (6-19), merged with the frontier seeds 0-5
            run(gates(0.2, cal, 'full', BASE, which=('random', 'vq', 'aq')), list(range(N_FRONT, N_MAIN)), 'rho02')
            jobs = []
            for hf in [1.5, 3.0]:
                jobs += [(k.replace('_0.1', f'_{hf}'), c, kw) for k, c, kw in gates(RHO, cal, 'full', BASE, hf=hf, which=('aq',))]
            run(jobs, list(range(N_HMAX)), 'hmax_aq')
        elif stage == 'dense_calib':
            with Pool(2) as P:
                C = P.map(calib_dense, [100, 101])
            save('dense_calib', fit_calibration(C, [FULL]))
        elif stage == 'dense':
            dc = load('dense_calib'); MAPS['dense'] = (dc['knots'][0], dc['vals'][0])
            jobs = [('s1', DENSE, ep(DENSE, gate='s1')), ('s2', DENSE, ep(DENSE, gate='s2'))]
            jobs += gates(RHO, dc, 'dense', DENSE, which=('random', 'static', 'vq'))
            jobs += [(k, c, kw) for k, c, kw in gates(0.3, dc, 'dense', DENSE, which=('vq',))]
            run(jobs, list(range(N_DENSE)), 'dense')
        print(stage, 'done', round(time.time() - t0), flush=True)
    print('all done', round(time.time() - t0), flush=True)
