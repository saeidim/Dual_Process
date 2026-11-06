"""Where does each budget-enforcing gate spend its deliberations?  Logs the signal
and the deliberation decisions of always-S1 and of the gates on realizations 0-5."""
import os, json, sys
import numpy as np
from multiprocessing import Pool
import run_revision as R
from p2_core import run_episode
from v2xsim import Config
cal = R.load('calib'); R.apply_tuning()
R.MAPS['full'] = (cal['knots'][1], cal['vals'][1])
def job(a):
    tag, kw, seed = a
    kw = dict(kw)
    if 'vmap_ref' in kw:
        kw['vmap'] = R.vm(*R.MAPS[kw.pop('vmap_ref')])
    r = run_episode(Config(**R.BASE), seed=seed, q=R.Q, record_v=True, **kw)
    v = np.array(r['vlog']['v']); z = np.array(r['vlog']['z'])
    thr = cal['thr']['0.1']['static']
    # z here flags slots in which a deliberation was run
    return tag, seed, dict(highfrac=float(np.mean(v >= thr)), n=int(len(v)), rate=float(r['s2_rate']),
                           frac_low=float(np.mean(v[z == 1] < thr)) if z.sum() else 0.0,
                           cover=float(np.mean(z[v >= thr])) if (v >= thr).any() else 0.0)
if __name__ == '__main__':
    jobs = [('s1', R.ep(R.BASE, gate='s1'))] + [(k, kw) for k, c, kw in R.gates(0.1, cal, 'full', R.BASE, which=('random', 'vq', 'aq', 'pd'))]
    allj = [(t, kw, s) for t, kw in jobs for s in range(6)]
    with Pool(2) as P:
        res = P.map(job, allj)
    out = {}
    for t, s, r in res:
        out.setdefault(t, []).append(r)
    summ = {t: {k: float(np.mean([r[k] for r in v])) for k in v[0]} for t, v in out.items()}
    json.dump(dict(per=out, mean=summ), open(os.path.join(R.OUT, 'burst_diag.json'), 'w'))
    print(json.dumps(summ, indent=1))
