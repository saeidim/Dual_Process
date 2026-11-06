"""Re-runs the virtual-queue experiments with the warm-started queue (final
algorithm) and merges them into results/main_rho.json and results/V_sweep.json."""
import os, json
import numpy as np
from multiprocessing import Pool
import run_p2_experiments as R
from p2_core import ValueMap

R.Q = tuple(json.load(open(os.path.join(R.OUT, 'sysid.json'))).values())
cal = json.load(open(os.path.join(R.OUT, 'calibration.json')))
VM = ValueMap(); VM.knots = np.array(cal['knots']); VM.vals = np.array(cal['vals']); R.VMAP = VM
THR = {float(k): v for k, v in cal['thr'].items()}


def run(jobs):
    allj = [(tag, dict(kw), s) for (tag, kw) in jobs for s in R.SEEDS]
    with Pool(2) as P:
        res = P.map(R.job, allj, chunksize=1)
    out = {}
    for tag, s, r in res:
        out.setdefault(tag, []).append(r)
    return out


if __name__ == '__main__':
    jobs = [(f'vq_{rho}', dict(gate='vq', rho=rho, V=50.0, Q0=50.0 * VM(THR[rho]['static']))) for rho in R.RHOS]
    p = os.path.join(R.OUT, 'main_rho.json'); d = json.load(open(p)); d.update(run(jobs)); json.dump(d, open(p, 'w'))
    print('main vq', flush=True)
    jobs = [(f'vq_V{V}', dict(gate='vq', rho=0.1, V=V, Q0=V * VM(THR[0.1]['static']), record_trace=True))
            for V in [0.5, 2.0, 10.0, 50.0, 250.0]]
    jobs += [('vq_V50.0_cold', dict(gate='vq', rho=0.1, V=50.0, Q0=0.0, record_trace=True))]
    json.dump(run(jobs), open(os.path.join(R.OUT, 'V_sweep.json'), 'w'))
    print('V done', flush=True)
