"""Diagnostic for Lemma 1: how much each world-model perturbation of System 2
changes the imagined delivery probabilities.  Along System-1 episodes we take
every 10th data slot, imagine M=64 rollouts of H=32 slots with the nominal and
with each perturbed model (common random numbers), and record the absolute
change of the rollout-averaged success probability of every short-list vehicle
in every imagined data slot."""
import os, sys, json
import numpy as np
from multiprocessing import Pool
from p2_core import *
from v2xsim import Config
import p2_core as P2
HERE = os.path.dirname(os.path.abspath(__file__))
Q = tuple(json.load(open(os.path.join(HERE, 'results', 'sysid.json'))).values())
VAR = {'q0.01': dict(q=0.01), 'q100': dict(q=100.0), 'nosf': dict(sf=0.0), 'noblock': dict(b=0.0)}


def one(seed):
    cfg = Config(n_veh=16, look_slots=8); tab = get_table(cfg)
    rng_env = np.random.default_rng(10_000 + seed); rng_ch = np.random.default_rng(30_000 + seed)
    from v2xsim import Highway, WorldModel, predict_success
    env = Highway(cfg, rng_env, rng_ch); K = cfg.n_veh; tau = cfg.look_slots
    wm = WorldModel(cfg, dict(ant_h=env.ant_h, body_h=env.body_h, length=env.length), q_x=Q[0], q_y=Q[1])
    for b in range(6):
        wm.update(env.sense())
        for _ in range(50):
            env.step(); wm.predict(1)
    h = np.ones(K); out = {k: [] for k in VAR}; next_look = 0; busy = 0; nd = 0
    for t in range(1500):
        if t == next_look:
            wm.update(env.sense()); next_look = t + 128; busy = tau
        if busy > 0:
            busy -= 1; h = h + 1
        else:
            ren = wm.render(); sm, ss = wm.sf_pred()
            p_now = np.where(wm.init, predict_success(ren['snr_nosf'], ren['J'], sm, ss, wm.sigma_u(ren), tab, cfg.N_ant), 0.0)
            idx1 = np.maximum(p_now, 1e-6) * h * (h + 2); a1 = int(np.argmax(idx1)); cands = np.argsort(-idx1)[:4]
            if nd % 10 == 0:
                rel = np.arange(32); lm = ((t + rel) >= next_look) & ((t + rel) < next_look + tau)
                base = imagine_samples(wm, 32, 64, tab, lm, np.random.default_rng(seed * 100000 + t)).mean(0)
                for k, d in VAR.items():
                    pp = imagine_samples(wm, 32, 64, tab, lm, np.random.default_rng(seed * 100000 + t), d.get('q', 1.0),
                                         d.get('b', 1.0), d.get('sf', 1.0)).mean(0)
                    out[k].append(np.abs(pp - base)[~lm][:, cands].ravel())
            nd += 1
            ok, _ = env.transmit(a1, ren['u'][a1]); h = h + 1
            if ok:
                h[a1] = 1
            else:
                wm.nack(a1, tab.snr_at(0.1))
        env.step(); wm.predict(1)
    return {k: np.concatenate(x).tolist() for k, x in out.items()}


if __name__ == '__main__':
    with Pool(2) as Pl:
        R = Pl.map(one, [0, 1, 2, 3])
    res = {}
    for k in VAR:
        x = np.concatenate([np.array(r[k]) for r in R])
        res[k] = dict(mean=float(x.mean()), p99=float(np.percentile(x, 99)), max=float(x.max()), n=int(len(x)))
    json.dump(res, open(os.path.join(HERE, 'results_rev', 'eps_diag.json'), 'w'))
    print(res)
