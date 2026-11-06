"""Dense scenario: virtual-queue gate at a 30% budget (merged into dense_eval.json)."""
import os, json
import numpy as np
import run_p2_expansion as X
from multiprocessing import Pool
if __name__ == '__main__':
    dc = json.load(open(os.path.join(X.OUT, 'dense_calib.json')))
    X.G['dense'] = (dc['knots'], dc['vals'])
    v = np.array(dc['v']); lam = X.lam_for_budget(v, np.interp(v, dc['knots'], dc['vals'])[:, None], [1.0], 0.3)
    res = X.run([('vq30', X.DENSE, dict(gate='vq', rho=0.3, V=X.V, vmap_ref='dense', Q0=X.V * lam))], list(range(6)))
    p = os.path.join(X.OUT, 'dense_eval.json'); d = json.load(open(p)); d.update(res); json.dump(d, open(p, 'w'))
    print('done', flush=True)
