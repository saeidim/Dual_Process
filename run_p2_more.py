"""Final pass: virtual-queue gate at its operating point V=10 (warm start), and five
more traffic realizations (seeds 3-7) for every gate at rho=0.1."""
import os, json
import numpy as np
from multiprocessing import Pool
import run_p2_experiments as R
from p2_core import ValueMap

R.Q = tuple(json.load(open(os.path.join(R.OUT, 'sysid.json'))).values())
cal = json.load(open(os.path.join(R.OUT, 'calibration.json')))
VM = ValueMap(); VM.knots = np.array(cal['knots']); VM.vals = np.array(cal['vals']); R.VMAP = VM
THR = {float(k): v for k, v in cal['thr'].items()}
V = 10.0


def run(jobs, seeds):
    allj = [(tag, dict(kw), s) for (tag, kw) in jobs for s in seeds]
    with Pool(2) as P:
        res = P.map(R.job, allj, chunksize=1)
    out = {}
    for tag, s, r in res:
        out.setdefault(tag, []).append(r)
    return out


def vq(rho):
    return dict(gate='vq', rho=rho, V=V, Q0=V * VM(THR[rho]['static']))


if __name__ == '__main__':
    p = os.path.join(R.OUT, 'main_rho.json'); d = json.load(open(p))
    new = run([(f'vq_{rho}', vq(rho)) for rho in R.RHOS], [0, 1, 2])
    d.update(new)
    extra_jobs = [('s1', dict(gate='s1')), ('s2', dict(gate='s2')),
                  ('random_0.1', dict(gate='random', rho=0.1)), ('periodic_0.1', dict(gate='periodic', rho=0.1)),
                  ('unc_0.1', dict(gate='unc', rho=0.1, unc_thr=THR[0.1]['unc'])),
                  ('static_0.1', dict(gate='static', rho=0.1, thr=THR[0.1]['static'])),
                  ('vq_0.1', vq(0.1))]
    extra = run(extra_jobs, [3, 4, 5, 6, 7])
    for k, v in extra.items():
        d[k] = d[k][:3] + v
    json.dump(d, open(p, 'w'))
    print('done', flush=True)
