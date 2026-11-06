"""Second-revision experiments (reviewer concerns on the cheap-planner baseline,
the strength of System 1, deliberation latency, world-model error, trace length
and learned gates).  Writes results_rev2/<stage>.json; a stage is skipped if its
file exists.  Usage: python run_revision2.py [stage ...]

Planners (compute cost c in units of the full planner, c = M H / 512):
  TINY = (M=2,  H=8,  c=1/32)    LITE = (M=4, H=16, c=1/8)    FULL = (M=16, H=32, c=1)
Budgets of single-planner gates are fractions of data slots in which that
planner is invoked; the compute per slot is that fraction times c.
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')         # single-threaded BLAS/OpenMP: safe with forked workers
import numpy as np
from multiprocessing import Pool
import run_revision as R
from p2_core import run_episode, ValueMap
from v2xsim import Config

HERE = os.path.dirname(os.path.abspath(__file__))
OUT2 = os.path.join(HERE, 'results_rev2'); os.makedirs(OUT2, exist_ok=True)
BASE = R.BASE
TINY = dict(M=2, H=8, c=1.0 / 32)
LITE = R.LITE; FULL = R.FULL
SEEDS = list(range(20)); CAL_SEEDS = [100, 101]; LONG_SEEDS = [200, 201, 202]
NS = 6000; NCAL = 4000; NLONG = 24000
MAPS = {}
if os.environ.get('SMOKE') == '1':                 # tiny end-to-end check of every stage
    OUT2 = os.path.join(HERE, 'results_smoke'); os.makedirs(OUT2, exist_ok=True)
    R.NS = NS = 200; NCAL = 150; NLONG = 300; SEEDS = [0, 1]; LONG_SEEDS = [200]


def p2(n):
    return os.path.join(OUT2, n + '.json')


def save2(n, o):
    json.dump(o, open(p2(n), 'w'))


def load2(n):
    return json.load(open(p2(n)))


class LearnedMap:
    """Value map on a feature vector (gradient-boosted regression of the
    reference-measured gain of the full planner), clipped at zero."""
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return max(float(self.model.predict(np.asarray(x, float)[None, :])[0]), 0.0)


def job(a):
    tag, kw, seed = a
    kw = dict(kw)
    if 'vmap_ref' in kw:
        kw['vmap'] = R.vm(*MAPS[kw.pop('vmap_ref')])
    if 'vmaps_ref' in kw:
        kw['vmaps'] = [R.vm(*MAPS[r], clip0=False) for r in kw.pop('vmaps_ref')]
    t = time.time()
    r = run_episode(Config(**BASE), seed=seed, q=R.Q, **kw)
    r['wall'] = time.time() - t; r['seed'] = seed
    r.pop('trace', None)
    if 'vlog' in r and not kw.get('keep_vlog'):
        v = np.array(r['vlog']['v']); r['vq'] = np.quantile(v, [0.7, 0.8, 0.9, 0.95]).tolist(); r.pop('vlog')
    return tag, seed, r


def run(jobs, seeds, name=None):
    allj = [(tag, kw, s) for (tag, kw) in jobs for s in seeds]
    w = lambda a: a[1].get('n_slots', NS) * (a[1].get('M', 16) * a[1].get('H', 32) / 512.0
                                             * (1.0 if a[1].get('gate') == 's2' else a[1].get('rho', 0.1)) + 0.05)
    allj.sort(key=w, reverse=True)
    out = {}
    with Pool(2) as P:
        for tag, s, r in P.imap_unordered(job, allj, chunksize=1):
            out.setdefault(tag, []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r['seed'])
    if name:
        save2(name, out)
    return out


def ep(**kw):
    return R.ep(BASE, **kw)


def calib_job(seed):
    return run_episode(Config(**BASE), gate='s1', seed=seed, q=R.Q, planners=[TINY, LITE, FULL], log_multi=True,
                       **ep(n_slots=NCAL))['mlog']


def lam_single(v, gmap, rho):
    g = np.maximum(np.interp(v, gmap[0], gmap[1]), 0.0)[:, None]
    return R.lam_for_budget(v, g, [1.0], rho)


STAGES = ['calib2', 'tune2', 'cheap', 'cs1', 'delay', 'tiny_gates', 'multi_low', 'learned', 'mismatch2', 'long', 'delay_sweep']

if __name__ == '__main__':
    want = sys.argv[1:] or STAGES
    t0 = time.time()
    R.apply_tuning()                                # step sizes of the first revision (V=1, eta=1e-3, eta0=100)
    cal1 = R.load('calib')
    MAPS['full'] = (cal1['knots'][1], cal1['vals'][1]); MAPS['lite'] = (cal1['knots'][0], cal1['vals'][0])
    for stage in STAGES:
        if stage not in ('calib2', 'tune2') and (stage not in want or os.path.exists(p2(stage))):
            continue
        if stage == 'calib2' and not os.path.exists(p2('calib2')):
            with Pool(2) as P:
                C = P.map(calib_job, CAL_SEEDS)
            v = np.concatenate([c['v'] for c in C]); g = np.concatenate([np.array(c['g']) for c in C])
            feat = np.concatenate([np.array(c['feat']) for c in C])
            maps = [ValueMap().fit(v, g[:, i]) for i in range(3)]
            save2('calib2', dict(v=v.tolist(), g=g.tolist(), feat=feat.tolist(),
                                 knots=[m.knots.tolist() for m in maps], vals=[m.vals.tolist() for m in maps]))
            print('calib2 done', round(time.time() - t0), flush=True)
        c2 = load2('calib2')
        MAPS['tiny2'] = (c2['knots'][0], c2['vals'][0]); MAPS['lite2'] = (c2['knots'][1], c2['vals'][1])
        MAPS['full2'] = (c2['knots'][2], c2['vals'][2])
        v2 = np.array(c2['v'])
        if stage == 'tune2' and not os.path.exists(p2('tune2')):
            # (a) anticipation of the cost-aware System 1, (b) step sizes of the gates with the tiny planner,
            # (c) V of the three-planner gate at the two compute budgets; all on the calibration realizations.
            jobs = [(f's1ca|{d}', ep(gate='s1', s1_delta=d, record_v=True)) for d in [0, 4, 8, 16]]
            for rho in [0.1, 0.2]:
                lam = lam_single(v2, MAPS['tiny2'], rho)
                for x in [0.1, 0.3, 1.0, 3.0, 10.0]:
                    jobs.append((f'vqT{rho}|{x}', ep(gate='vq', rho=rho, V=x, vmap_ref='tiny2', Q0=x * lam, **{k: TINY[k] for k in 'MH'})))
                thr = float(np.quantile(v2, 1 - rho))
                for x in [1e-4, 1e-3, 1e-2]:
                    jobs.append((f'aqT{rho}|{x}', ep(gate='aq', rho=rho, thr=thr, eta=x, **{k: TINY[k] for k in 'MH'})))
            gm = np.stack([np.interp(v2, c2['knots'][i], c2['vals'][i]) for i in range(3)], 1)
            for rb in [1 / 320.0, 1 / 160.0]:
                lamm = R.lam_for_budget(v2, gm, [TINY['c'], LITE['c'], FULL['c']], rb)
                for x in [0.03, 0.1, 0.3, 1.0]:
                    jobs.append((f'multi{rb:.5f}|{x}', ep(gate='multi', rho=rb, V=x, planners=[TINY, LITE, FULL],
                                                          vmaps_ref=['tiny2', 'lite2', 'full2'], Q0=x * lamm)))
            res = run(jobs, CAL_SEEDS)
            table = {k: dict(cost=float(np.mean([r['cost'] for r in v])), rate=float(np.mean([r['compute_rate'] for r in v])),
                             s2=float(np.mean([r['s2_rate'] for r in v])),
                             vq=np.mean([r.get('vq', [0, 0, 0, 0]) for r in v], 0).tolist()) for k, v in res.items()}
            ch = {}
            ch['delta'] = min([0, 4, 8, 16], key=lambda d: table[f's1ca|{d}']['cost'])
            ch['s1ca_vq'] = table[f"s1ca|{ch['delta']}"]['vq']          # quantiles 0.7, 0.8, 0.9, 0.95 of v under S1-CA
            for g_, grid in [('vqT', [0.1, 0.3, 1.0, 3.0, 10.0]), ('aqT', [1e-4, 1e-3, 1e-2])]:
                for rho in [0.1, 0.2]:
                    cand = [(x, table[f'{g_}{rho}|{x}']['s2'], table[f'{g_}{rho}|{x}']['cost']) for x in grid]
                    ok = [c for c in cand if c[1] <= 1.05 * rho]
                    ch[f'{g_}{rho}'] = min(ok, key=lambda c: c[2])[0] if ok else min(cand, key=lambda c: abs(c[1] - rho))[0]
            for rb in [1 / 320.0, 1 / 160.0]:
                grid = [0.03, 0.1, 0.3, 1.0]
                cand = [(x, table[f'multi{rb:.5f}|{x}']['rate'], table[f'multi{rb:.5f}|{x}']['cost']) for x in grid]
                ok = [c for c in cand if c[1] <= 1.05 * rb]
                ch[f'multi{rb:.5f}'] = min(ok, key=lambda c: c[2])[0] if ok else min(cand, key=lambda c: abs(c[1] - rb))[0]
            save2('tune2', dict(chosen=ch, table=table))
            print('tune2', ch, flush=True)
            continue
        if stage in ('calib2', 'tune2'):
            continue
        ch = load2('tune2')['chosen']
        D = ch['delta']; TM = {k: TINY[k] for k in 'MH'}
        thr = {rho: float(np.quantile(v2, 1 - rho)) for rho in [0.1, 0.2]}
        thr_ca = {0.1: ch['s1ca_vq'][2], 0.2: ch['s1ca_vq'][1]}
        if stage == 'cheap':
            jobs = [('s2_tiny', ep(gate='s2', record_timing=True, **TM)),
                    ('s2_m2h16', ep(gate='s2', M=2, H=16)),
                    ('s2_lite', ep(gate='s2', M=4, H=16, record_timing=True))]
            res = run(jobs[:2], SEEDS)
            res.update(run(jobs[2:], list(range(10, 20))))
            old = R.load('multi')['s2_lite']                     # realizations 0-9 from the first revision
            res['s2_lite'] = sorted(old + res['s2_lite'], key=lambda r: r['seed'])
            save2('cheap', res)
        elif stage == 'cs1':
            jobs = [('s1ca', ep(gate='s1', s1_delta=D)),
                    ('s2ca', ep(gate='s2', s1_delta=D)),
                    ('s2ca_tiny', ep(gate='s2', s1_delta=D, **TM)),
                    ('aqca_0.2', ep(gate='aq', rho=0.2, thr=thr_ca[0.2], eta=R.ETA_AQ, s1_delta=D)),
                    ('aqca_tiny_0.2', ep(gate='aq', rho=0.2, thr=thr_ca[0.2], eta=ch['aqT0.2'], s1_delta=D, **TM))]
            run(jobs, SEEDS, 'cs1')
        elif stage == 'delay':
            ch_ = load2('cheap')
            t_full = 40.0                                           # median latency on the authors' machine [ms]
            # tiny-planner latency scaled by the ratio measured here (same code, same machine for both)
            ref = run_episode(Config(**BASE), seed=0, q=R.Q, **ep(gate='s2', record_timing=True, n_slots=min(600, NS)))
            r_ = np.median([r['timing']['s2'][0] for r in ch_['s2_tiny']]) / ref['timing']['s2'][0]
            d_tiny = int(np.ceil(t_full * r_))
            jobs = [('s2_d40', ep(gate='s2', s2_delay=40)),
                    ('aq_0.2_d40', ep(gate='aq', rho=0.2, thr=thr[0.2], eta=R.ETA_AQ, s2_delay=40)),
                    (f's2tiny_d{d_tiny}', ep(gate='s2', s2_delay=d_tiny, **TM)),
                    (f's2tiny_d{d_tiny}_ahead', ep(gate='s2', s2_delay=d_tiny, s2_ahead=True, **TM)),
                    (f'aqtiny_0.2_d{d_tiny}', ep(gate='aq', rho=0.2, thr=thr[0.2], eta=ch['aqT0.2'], s2_delay=d_tiny, **TM))]
            res = run(jobs, SEEDS)
            res['_meta'] = dict(ratio=float(r_), d_tiny=d_tiny, t_full_here_ms=1e3 * ref['timing']['s2'][0])
            save2('delay', res)
        elif stage == 'tiny_gates':
            jobs = []
            for rho in [0.1, 0.2]:
                lam = lam_single(v2, MAPS['tiny2'], rho); Vt = ch[f'vqT{rho}']
                jobs += [(f'random_{rho}', ep(gate='random', rho=rho, **TM)),
                         (f'static_{rho}', ep(gate='static', rho=rho, thr=thr[rho], **TM)),
                         (f'aq_{rho}', ep(gate='aq', rho=rho, thr=thr[rho], eta=ch[f'aqT{rho}'], **TM)),
                         (f'vq_{rho}', ep(gate='vq', rho=rho, V=Vt, vmap_ref='tiny2', Q0=Vt * lam, **TM))]
            run(jobs, SEEDS, 'tiny_gates')
        elif stage == 'multi_low':
            gm = np.stack([np.interp(v2, c2['knots'][i], c2['vals'][i]) for i in range(3)], 1)
            jobs = []
            for rb in [1 / 320.0, 1 / 160.0]:
                lamm = R.lam_for_budget(v2, gm, [TINY['c'], LITE['c'], FULL['c']], rb); x = ch[f'multi{rb:.5f}']
                jobs.append((f'multi_{rb:.5f}', ep(gate='multi', rho=rb, V=x, planners=[TINY, LITE, FULL],
                                                   vmaps_ref=['tiny2', 'lite2', 'full2'], Q0=x * lamm)))
            run(jobs, SEEDS, 'multi_low')
        elif stage == 'learned':
            from sklearn.ensemble import HistGradientBoostingRegressor
            X = np.array(c2['feat']); y = np.array(c2['g'])[:, 2]
            mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                min_samples_leaf=40, random_state=0).fit(X, y)
            lm = LearnedMap(mdl)
            pred = np.maximum(mdl.predict(X), 0.0)
            lam = R.lam_for_budget(v2, pred[:, None], [1.0], 0.1)
            thr_l = float(np.quantile(pred, 0.9))
            # capture of the measured gain on calibration data (in-sample for the learned map; reported as such)
            from scipy.stats import spearmanr
            jobs = [('vq_learned', ep(gate='vq', rho=0.1, V=R.V, vmap=lm, vfeat='learned', Q0=R.V * lam)),
                    ('pd_learned', ep(gate='pd', rho=0.1, vmap=lm, vfeat='learned', lam0=lam, eta0=R.ETA0_PD))]
            res = run(jobs, SEEDS)
            res['_meta'] = dict(lam=float(lam), thr=thr_l, spearman_learned=float(spearmanr(pred, y)[0]),
                                spearman_v=float(spearmanr(v2, y)[0]))
            save2('learned', res)
        elif stage == 'mismatch2':
            jobs = [(f'pl{b:+g}', ep(gate='s2', s2_pl=b, n_slots=NS // 2)) for b in [-12.0, -6.0, 6.0, 12.0]]
            jobs += [('block3', ep(gate='s2', s2_block=3.0, n_slots=NS // 2))]
            run(jobs, list(range(6)), 'mismatch2')
        elif stage == 'long':
            jobs = [('s1', ep(gate='s1', n_slots=NLONG)),
                    ('s2', ep(gate='s2', n_slots=NLONG)),
                    ('s2_tiny', ep(gate='s2', n_slots=NLONG, **TM)),
                    ('aq_0.2', ep(gate='aq', rho=0.2, thr=thr[0.2], eta=R.ETA_AQ, n_slots=NLONG)),
                    ('random_0.2', ep(gate='random', rho=0.2, n_slots=NLONG))]
            run(jobs, LONG_SEEDS, 'long')
        elif stage == 'delay_sweep':
            # how short must the latency be?  tiny planner in every slot, plans applied d data slots later
            run([(f's2tiny_d{d}', ep(gate='s2', s2_delay=d, **TM)) for d in [1, 2, 4]], SEEDS, 'delay_sweep')
        print(stage, 'done', round(time.time() - t0), flush=True)
    print('all done', round(time.time() - t0), flush=True)
