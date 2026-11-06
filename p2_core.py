"""
p2_core.py -- "Thinking fast and slow at the network edge": budget-constrained
metacognitive deliberation for world-model-based AoI scheduling.

System 1 (fast)  : max-weight index  beta_k * p_k * h_k * (h_k + 2)  on the
                   world model's current success predictions.
System 2 (slow)  : Monte-Carlo rollout planning in the world model.  M belief
                   samples of all vehicles are imagined H slots ahead (CWNA
                   kinematics + physics renderer); every candidate first action
                   is scored by the discounted weighted AoI of following System 1
                   afterwards (common random numbers), and the best is taken.
Metacognitive gate (decides, every data slot, whether to invoke System 2):
  's1'      never        |  's2'     always
  'random'  w.p. rho     |  'periodic' every round(1/rho) slots
  'unc'     belief-uncertainty threshold (calibrated to rate rho)
  'oracle'  deliberates only when the realised gain is in the top rho (needs S2
            every slot; reference upper bound for any proxy-based gate)
  'static'  threshold on the predicted value of deliberation g_hat(v)
            (Theorem 4, optimal stationary gate; threshold set offline to rate rho)
  'vq'      proposed online virtual-queue gate: z = 1{V g_hat(v) >= Q},
            Q <- max(Q + z - rho, 0)      (Theorem 5)
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2xsim import (Config, Highway, WorldModel, SuccessTable, predict_success,
                    wrap, geometry, blockage_loss, mean_snr_dB, array_gain)

_TABLES = {}


def get_table(cfg):
    key = (cfg.r_req, cfg.F_sub, cfg.K_rice_dB)
    if key not in _TABLES:
        _TABLES[key] = SuccessTable(cfg)
    return _TABLES[key]


# ----------------------------------------------------------------------------
# Imagination (shared by the gate proxy and by System 2)
# ----------------------------------------------------------------------------
def imagine_mean(wm, H, tab, look_mask, block_scale=1.0, sf_scale=1.0, pl_bias=0.0):
    """Mean-trajectory imagination: success predictions p_bar[tau, k] for the
    next H slots when steering at the imagined mean (cheap: one rollout)."""
    cfg = wm.cfg; D = cfg.slot
    taus = np.arange(H)[:, None] * D
    m = np.broadcast_to(wm.m, (H,) + wm.m.shape).copy()
    m[..., 0] = m[..., 0] + m[..., 1] * taus
    m[..., 2] = m[..., 2] + m[..., 3] * taus
    xr = wrap(m[..., 0], cfg.L_road); y = m[..., 2]
    ah = np.broadcast_to(wm.ant_h, xr.shape)
    d3, u = geometry(xr, y, ah, cfg)
    J = block_scale * blockage_loss(xr, y, ah, np.broadcast_to(wm.body_h, xr.shape), np.broadcast_to(wm.length, xr.shape), cfg)
    snr = mean_snr_dB(d3, J, 0.0, cfg) - pl_bias
    p = np.zeros((H, cfg.n_veh))
    for i in range(H):
        mu, sd = wm.sf_pred(extra_dist=np.abs(wm.m[:, 1]) * D * i)
        sd = sf_scale * sd
        a = (wm.age + i).astype(float)
        # belief std of the direction cosine at age a (CWNA closed form)
        su = _sigma_u_age(wm, xr[i], y[i], i)
        p[i] = predict_success(snr[i], J[i], mu, sd, su, tab, cfg.N_ant)
    p[:, ~wm.init] = 0.0
    p[look_mask] = 0.0
    return p, u


def _sigma_u_age(wm, xr, y, extra):
    cfg = wm.cfg; t = extra * cfg.slot
    vx = wm.Px[:, 0, 0] + 2 * t * wm.Px[:, 0, 1] + t * t * wm.Px[:, 1, 1] + wm.q_x * t ** 3 / 3
    vy = wm.Py[:, 0, 0] + 2 * t * wm.Py[:, 0, 1] + t * t * wm.Py[:, 1, 1] + wm.q_y * t ** 3 / 3
    dy = y - cfg.y_rsu; dh = wm.ant_h - cfg.h_rsu
    d = np.sqrt(xr ** 2 + dy ** 2 + dh ** 2)
    Jx = (d ** 2 - xr ** 2) / d ** 3; Jy = -xr * dy / d ** 3
    return np.sqrt(np.maximum(Jx ** 2 * vx + Jy ** 2 * vy, 0))


def imagine_samples(wm, H, M, tab, look_mask, rng, qscale=1.0, block_scale=1.0, sf_scale=1.0, pl_bias=0.0):
    """Sampled imagination: M joint belief samples propagated H slots with CWNA
    process noise; returns success probabilities ps[m, tau, k] when the beam is
    steered at the imagined mean (as System 1 would)."""
    cfg = wm.cfg; D = cfg.slot; K = cfg.n_veh
    # sample initial states from the Gaussian beliefs
    Lx = np.linalg.cholesky(wm.Px + 1e-12 * np.eye(2)); Ly = np.linalg.cholesky(wm.Py + 1e-12 * np.eye(2))
    ex = np.einsum('kij,mkj->mki', Lx, rng.standard_normal((M, K, 2)))
    ey = np.einsum('kij,mkj->mki', Ly, rng.standard_normal((M, K, 2)))
    st = np.empty((M, K, 4))
    st[..., 0] = wm.m[:, 0] + ex[..., 0]; st[..., 1] = wm.m[:, 1] + ex[..., 1]
    st[..., 2] = wm.m[:, 2] + ey[..., 0]; st[..., 3] = wm.m[:, 3] + ey[..., 1]
    # CWNA increments (exact discretisation per slot)
    qx, qy = wm.q_x * qscale, wm.q_y * qscale
    Qd = np.array([[D ** 3 / 3, D ** 2 / 2], [D ** 2 / 2, D]])
    Lq = np.linalg.cholesky(Qd)
    X = np.empty((M, H, K)); Y = np.empty((M, H, K))
    for i in range(H):
        X[:, i] = st[..., 0]; Y[:, i] = st[..., 2]
        wx = np.sqrt(qx) * (rng.standard_normal((M, K, 2)) @ Lq.T)
        wy = np.sqrt(qy) * (rng.standard_normal((M, K, 2)) @ Lq.T)
        st[..., 0] += st[..., 1] * D + wx[..., 0]; st[..., 1] += wx[..., 1]
        st[..., 2] += st[..., 3] * D + wy[..., 0]; st[..., 3] += wy[..., 1]
    xr = wrap(X, cfg.L_road)
    ah = np.broadcast_to(wm.ant_h, xr.shape)
    d3, u = geometry(xr, Y, ah, cfg)
    J = block_scale * blockage_loss(xr, Y, ah, np.broadcast_to(wm.body_h, xr.shape), np.broadcast_to(wm.length, xr.shape), cfg)
    # steering at the mean trajectory
    _, u_mean = imagine_mean_u(wm, H)
    g = array_gain(u - u_mean[None], cfg.N_ant)
    ps = np.zeros((M, H, K))
    for i in range(H):
        mu, sd = wm.sf_pred(extra_dist=np.abs(wm.m[:, 1]) * D * i)
        sf = mu[None] + sf_scale * sd[None] * rng.standard_normal((M, K))
        snr = mean_snr_dB(d3[:, i], J[:, i], sf, cfg) - pl_bias + 10 * np.log10(np.maximum(g[:, i], 1e-12))
        ps[:, i] = tab(snr, J[:, i] > 3.0)
    ps[:, :, ~wm.init] = 0.0
    ps[:, look_mask] = 0.0
    return ps


def imagine_mean_u(wm, H):
    cfg = wm.cfg; D = cfg.slot
    taus = np.arange(H)[:, None] * D
    x = wm.m[None, :, 0] + wm.m[None, :, 1] * taus
    y = wm.m[None, :, 2] + wm.m[None, :, 3] * taus
    xr = wrap(x, cfg.L_road)
    d3, u = geometry(xr, y, np.broadcast_to(wm.ant_h, xr.shape), cfg)
    return d3, u


def phi_maxweight(h):
    """Age weight of the classical max-weight AoI index (drift of sum_k h_k^2)."""
    return h * (h + 2)


def make_phi_cost(w, psi, h_max, delta):
    """Age weight of the cost-aware index (revision): the reduction of the
    cumulative-cost potential Phi(h) = sum_{i<=h} f(i) obtained by a delivery,
    Phi(h+1) - Phi(1), for the per-slot cost f(i) = w i + psi 1{i > h_max - delta}.
    delta >= 0 makes the index anticipate the tolerable age by delta slots."""
    thr = max(h_max - delta, 1.0)
    def phi(h):
        return w * ((h + 1) * (h + 2) / 2.0 - 1.0) + psi * np.maximum(h + 1 - thr, 0.0)
    return phi


def rollout_values(h0, beta, cands, ps, p_base, gamma, rng, h_max=np.inf, psi=0.0, phi=phi_maxweight, offset=0,
                   pre=None):
    """Discounted weighted-AoI cost of (first action c, then System 1 on the
    mean belief p_base) under each imagined sample (common random numbers).
    h0 (K,), cands (C,), ps (M,H,K), p_base (H,K).  Returns Q (C, M)."""
    M, H, K = ps.shape
    C = len(cands)
    U = rng.random((M, H))
    h = np.broadcast_to(h0, (C, M, K)).copy()
    cost = np.zeros((C, M))
    disc = 1.0
    ar = np.arange(K)
    for i in range(H):
        idx = beta[None, None, :] * np.maximum(p_base[i], 1e-6)[None, None, :] * phi(h)
        a_s1 = np.argmax(idx, axis=-1)
        if i == offset:
            ca = np.asarray(cands)[:, None]
            # candidate -1 ("defer"): System 1 decides in the branch slot
            a = np.where(ca >= 0, np.broadcast_to(ca, (C, M)), a_s1)
        else:
            a = a_s1
            if pre is not None and i < offset and pre[i] >= 0:
                a = np.full((C, M), int(pre[i]))          # action already committed for this slot
        succ_p = np.take_along_axis(np.broadcast_to(ps[:, i, :], (C, M, K)), a[..., None], axis=-1)[..., 0]
        succ = U[None, :, i] < succ_p
        data_slot = p_base[i].max() > 0           # look slots carry no data
        h = h + 1
        if data_slot:
            hit = (ar[None, None, :] == a[..., None]) & succ[..., None]
            h = np.where(hit, 1.0, h)
        cost += disc * ((h * beta[None, None, :]).sum(-1) + psi * (h > h_max).sum(-1)) / K
        disc *= gamma
    return cost


# ----------------------------------------------------------------------------
# Gate proxy (cheap metacognitive signal) and its learned value map
# ----------------------------------------------------------------------------
def proxy_signal(p_now, cands):
    """v_t = 1 - min_{k in C_t} p_hat_k: the risk that System 1's short list
    contains a link the world model is unsure about.  It reuses System 1's own
    success predictions, so the metacognitive signal costs nothing extra."""
    return float(1.0 - np.min(p_now[cands]))


def features(v, p_now, cands, h, h_max, idx1, wm, ren, cfg):
    """Cheap per-slot features for a learned value map (revision 2): all are by-
    products of System 1.  [v, p of System 1's choice, index margin (2nd/1st),
    max age in the short list / h_max, number of short-list vehicles with
    predicted blockage J > 3 dB, belief uncertainty]."""
    c = np.asarray(cands)
    srt = np.sort(idx1)[::-1]
    unc = float(np.sum(wm.sigma_u(ren) * cfg.N_ant * (idx1 / idx1.max())))
    return np.array([v, float(p_now[c[0]]), float(srt[1] / max(srt[0], 1e-12)),
                     float(h[c].max() / h_max), float(np.sum(ren['J'][c] > 3.0)), unc])


class ValueMap:
    """Monotone (isotonic, pool-adjacent-violators) regression g_hat(v) of the
    realised deliberation gain on the proxy, fitted on calibration logs."""
    def fit(self, v, g):
        o = np.argsort(v); v = np.asarray(v)[o]; g = np.asarray(g)[o]
        # PAVA
        vals = list(g.astype(float)); wts = [1.0] * len(vals); xs = [[x] for x in v]
        i = 0
        while i < len(vals) - 1:
            if vals[i] > vals[i + 1] + 1e-15:
                nv = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / (wts[i] + wts[i + 1])
                vals[i] = nv; wts[i] += wts[i + 1]; xs[i] += xs[i + 1]
                del vals[i + 1]; del wts[i + 1]; del xs[i + 1]
                i = max(i - 1, 0)
            else:
                i += 1
        self.knots = np.array([np.mean(x) for x in xs]); self.vals = np.array(vals)
        return self

    def __call__(self, v):
        return float(np.interp(v, self.knots, self.vals))


# ----------------------------------------------------------------------------
# Episode
# ----------------------------------------------------------------------------
def run_episode(cfg: Config, gate='s1', rho=0.1, V=50.0, seed=0, n_slots=12000,
                T_look=64, H=32, M=16, n_cand=4, gamma=0.97, q=(0.25, 0.17),
                vmap: ValueMap = None, thr=None, unc_thr=None, log_calib=False,
                record_trace=False, h_max_fac=3.0, psi=None, Q0=0.0,
                planners=None, vmaps=None, log_multi=False, M_ref=32, H_ref=32, s2_qscale=1.0,
                s2_block=1.0, s2_sf=1.0, eta=0.01, eta0=1.0, lam0=0.0, record_timing=False, record_v=False,
                s1_delta=None, s2_delay=0, s2_pl=0.0, vfeat=None, s2_ahead=False):
    """planners: list of dicts {M, H, c} (System 2 variants with compute cost c,
    in units of the full planner) used by gate='multi'; vmaps: their value maps.
    log_multi: evaluate every planner and an independent reference planner
    (M_ref rollouts, horizon H_ref) in every slot and log reference-measured gains.
    Additional gates (revision):
      'aq' : online quantile tracking on the raw signal, z = 1{v > theta},
             theta <- clip(theta + eta (z - rho), 0, 1), theta_0 = thr
      'pd' : Lagrangian (primal-dual) gate with diminishing dual step,
             z = 1{g_hat(v) > lam}, lam <- max(lam + eta0/sqrt(n+1) (z - rho), 0), lam_0 = lam0
    s2_block / s2_sf: System 2 imagines the knife-edge loss and the shadowing
    standard deviation scaled by these factors (world-model perturbations).
    Revision 2:
      s1_delta : None -> System 1 is the max-weight index h(h+2); a number ->
                 cost-aware index make_phi_cost(1, psi, h_max, s1_delta), used by
                 System 1 and by the imagined continuation of System 2;
      s2_delay : a deliberation started in slot t from the beliefs of slot t is
                 applied s2_delay data slots later (pipelined System 2); in the
                 meantime System 1 acts.  A stale plan is dropped if its vehicle
                 was served in between (System 1 acts instead);
      s2_ahead : with s2_delay = d > 0, System 2 plans for the slot in which its
                 decision takes effect: it imagines H + d slots, System 1 acts in
                 the first d imagined slots and the candidates branch at slot d
                 (offset rollout), and the plan is always applied; without it the
                 plan is stale by d slots (and dropped as above);
      s2_pl    : System 2 imagines s2_pl dB of extra path loss (biased world model);
      vfeat    : None -> gate signal v_t; 'learned' -> vmap is called on a
                 feature vector (learned value map, see features())."""
    import time as _time
    rng_env = np.random.default_rng(10_000 + seed)
    rng_ch = np.random.default_rng(30_000 + seed)
    rng_s2 = np.random.default_rng(40_000 + seed)
    rng_g = np.random.default_rng(50_000 + seed)
    env = Highway(cfg, rng_env, rng_ch)
    tab = get_table(cfg)
    K = cfg.n_veh; tau = cfg.look_slots
    wm = WorldModel(cfg, dict(ant_h=env.ant_h, body_h=env.body_h, length=env.length), q_x=q[0], q_y=q[1])
    for b in range(6):
        wm.update(env.sense())
        for _ in range(50):
            env.step(); wm.predict(1)
    h = np.ones(K); beta = np.ones(K)
    phi = phi_maxweight if s1_delta is None else make_phi_cost(1.0, (float(K) if psi is None else psi),
                                                                 h_max_fac * K, float(s1_delta))
    pending = {}                               # data-slot index -> (action, slot when planned)
    last_served = np.full(K, -10 ** 9)
    n_stale = 0; n_applied = 0
    hmax_list = np.array([2, 3, 4, 6, 8]) * K
    h_max = h_max_fac * K                      # maximum tolerable age (slots)
    psi = float(K) if psi is None else psi     # violation penalty per vehicle-slot
    st = dict(aoi=0.0, cost=0.0, vh=0.0, viol=np.zeros(len(hmax_list)), n_s2=0, n_data=0, n_succ=0, n_diff=0, peak=0.0,
              compute=0.0, n_j=np.zeros(1 + (len(planners) if planners else 0)))
    mlog = dict(v=[], g=[], gself=[], unc=[], feat=[]) if log_multi else None
    tim = dict(s1=[], s2=[]) if record_timing else None
    vlog = dict(v=[], z=[]) if record_v else None
    theta = float(thr) if (gate == 'aq' and thr is not None) else 0.5
    lam = float(lam0)
    qmax = 0.0

    def plan(Mj, Hj, cands_, rng_):
        rel_ = np.arange(Hj)
        lm = ((t + rel_) >= next_look) & ((t + rel_) < next_look + tau)
        pm, _ = imagine_mean(wm, Hj, tab, lm, s2_block, s2_sf, s2_pl)
        ps_ = imagine_samples(wm, Hj, Mj, tab, lm, rng_, s2_qscale, s2_block, s2_sf, s2_pl)
        Qv_ = rollout_values(h, beta, cands_, ps_, pm, gamma, rng_, h_max, psi, phi).mean(axis=1)
        return int(np.argmin(Qv_)), Qv_
    calib = dict(v=[], g=[], unc=[]) if log_calib else None
    trace = dict(z_avg=[], Q=[]) if record_trace else None
    Q = float(Q0)                              # warm start of the virtual queue
    period = max(int(round(1.0 / max(rho, 1e-9))), 1)
    next_look = 0; busy = 0
    for t in range(n_slots):
        if t == next_look:
            wm.update(env.sense())
            next_look = t + T_look; busy = tau
        if busy > 0:
            busy -= 1
            h = h + 1
        else:
            # ---- System 1 (always computed: it is the fast default) ----
            _t1 = _time.perf_counter()
            ren = wm.render(); sf_mu, sf_sd = wm.sf_pred()
            p_now = predict_success(ren['snr_nosf'], ren['J'], sf_mu, sf_sd, wm.sigma_u(ren), tab, cfg.N_ant)
            p_now = np.where(wm.init, p_now, 0.0)
            idx1 = beta * np.maximum(p_now, 1e-6) * phi(h)
            a1 = int(np.argmax(idx1))
            # slots of the horizon that fall inside the next look
            rel = np.arange(H)
            look_mask = ((t + rel) >= next_look) & ((t + rel) < next_look + tau)
            cands = list(np.argsort(-idx1)[:n_cand])
            # metacognitive signal (free: reuses System 1's predictions)
            v = proxy_signal(p_now, cands)
            if vfeat is not None or log_multi:
                feat = features(v, p_now, cands, h, h_max, idx1, wm, ren, cfg)
            vin = feat if vfeat is not None else v
            if record_timing:
                tim['s1'].append(_time.perf_counter() - _t1)
            # ---- metacognitive gate ----
            if gate == 's1':
                z = 0
            elif gate == 's2':
                z = 1
            elif gate == 'random':
                z = int(rng_g.random() < rho)
            elif gate == 'periodic':
                z = int(st['n_data'] % period == 0)
            elif gate == 'unc':
                z = int(np.sum(wm.sigma_u(ren) * cfg.N_ant * (idx1 / idx1.max())) >= unc_thr)
            elif gate == 'static':
                z = int(v >= thr)                 # g_hat is monotone in v: same ordering
            elif gate == 'vq':
                z = int(V * vmap(vin) > Q)
            elif gate == 'aq':
                z = int(v > theta)
            elif gate == 'pd':
                z = int(vmap(vin) > lam)
            elif gate == 'oracle':
                z = -1                            # decided after System 2 below
            elif gate == 'multi':
                scores = [0.0] + [V * vmaps[i](vin) - Q * planners[i]['c'] for i in range(len(planners))]
                jsel = int(np.argmax(scores))
                z = 0
            else:
                raise ValueError(gate)
            a = a1
            if log_multi:
                ref_i, Qref = plan(M_ref, H_ref, cands, rng_g)
                gj = []; gs = []
                for pl in planners:
                    ij, Qj = plan(pl['M'], pl['H'], cands, rng_s2)
                    gj.append(float(Qref[0] - Qref[ij])); gs.append(float(Qj[0] - Qj.min()))
                mlog['v'].append(v); mlog['g'].append(gj); mlog['gself'].append(gs)
                mlog['unc'].append(float(np.sum(wm.sigma_u(ren) * cfg.N_ant * (idx1 / idx1.max()))))
                mlog['feat'].append([float(x) for x in feat])
            if gate == 'multi':
                st['n_j'][jsel] += 1
                if jsel > 0:
                    pl = planners[jsel - 1]
                    ij, _ = plan(pl['M'], pl['H'], cands, rng_s2)
                    a = int(cands[ij]); st['n_s2'] += 1; st['n_diff'] += int(a != a1)
                    st['compute'] += pl['c']
                    Q = max(Q + pl['c'] - rho, 0.0)
                else:
                    Q = max(Q - rho, 0.0)
            if (z or log_calib or gate == 'oracle') and gate != 'multi':
                _t2 = _time.perf_counter()
                off = int(s2_delay) if s2_ahead else 0
                Hi = H + off
                lmi = look_mask if off == 0 else (((t + np.arange(Hi)) >= next_look) & ((t + np.arange(Hi)) < next_look + tau))
                p_mean, _ = imagine_mean(wm, Hi, tab, lmi, s2_block, s2_sf, s2_pl)
                ps = imagine_samples(wm, Hi, M, tab, lmi, rng_s2, s2_qscale, s2_block, s2_sf, s2_pl)
                pre = None
                if off > 0:
                    # actions already committed (pending plans) for the slots before the branch point
                    pre = np.full(off, -1); kd = st['n_data']
                    for i_ in range(off):
                        if not lmi[i_]:
                            pa_ = pending.get(kd)
                            pre[i_] = -1 if pa_ is None else pa_[0]
                            kd += 1
                cands_x = cands if off == 0 else list(cands) + [-1]
                Qv = rollout_values(h, beta, cands_x, ps, p_mean, gamma, rng_s2, h_max, psi, phi, offset=off,
                                    pre=pre).mean(axis=1)
                if record_timing:
                    tim['s2'].append(_time.perf_counter() - _t2)
                a2 = int(cands_x[int(np.argmin(Qv))])
                if log_calib:
                    srt = np.sort(idx1)[::-1]
                    calib['v'].append(v); calib['g'].append(float(Qv[0] - Qv.min()))
                    calib.setdefault('margin', []).append(float(srt[1] / srt[0]))
                    calib.setdefault('pmin', []).append(float(p_now[cands].min()))
                    calib['unc'].append(float(np.sum(wm.sigma_u(ren) * cfg.N_ant * (idx1 / idx1.max()))))
                if gate == 'oracle':
                    z = int(Qv[0] - Qv.min() > thr)
                if z:
                    st['n_s2'] += 1; st['compute'] += 1.0
                    if s2_delay > 0:
                        if a2 >= 0:                       # a2 = -1: defer to System 1 in that slot
                            pending[st['n_data'] + int(s2_delay)] = (a2, t)
                    else:
                        a = a2; st['n_diff'] += int(a2 != a1)
            if gate == 'vq':
                Q = max(Q + z - rho, 0.0)
            elif gate == 'aq':
                theta = min(max(theta + eta * (z - rho), 0.0), 1.0)
            elif gate == 'pd':
                lam = max(lam + eta0 / np.sqrt(st['n_data'] + 1.0) * (z - rho), 0.0)
            st['n_data'] += 1
            qmax = max(qmax, Q)
            if record_v:
                vlog['v'].append(float(v)); vlog['z'].append(int(jsel > 0) if gate == 'multi' else int(z == 1))
            if s2_delay > 0 and gate != 'multi':
                pl_ = pending.pop(st['n_data'] - 1, None)      # n_data was incremented above
                if pl_ is not None:
                    a_p, t_p = pl_
                    if (not s2_ahead) and last_served[a_p] >= t_p:
                        n_stale += 1                         # vehicle served meanwhile: plan dropped
                    else:
                        a = int(a_p); n_applied += 1; st['n_diff'] += int(a != a1)
            ok, info = env.transmit(a, ren['u'][a])
            st['n_succ'] += ok
            h = h + 1
            if ok:
                h[a] = 1; last_served[a] = t
            else:
                wm.nack(a, tab.snr_at(0.1))
            if record_trace:
                trace['z_avg'].append(st['compute'] / st['n_data']); trace['Q'].append(Q)
        st['aoi'] += np.sum(beta * h) / K
        st['cost'] += (np.sum(beta * h) + psi * np.sum(h > h_max)) / K
        st['vh'] += np.mean(h > h_max)
        st['viol'] += (h[None, :] > hmax_list[:, None]).mean(axis=1)
        st['peak'] = max(st['peak'], float(h.max()))
        env.step(); wm.predict(1)
    out = dict(aoi=st['aoi'] / n_slots, cost=st['cost'] / n_slots, viol_hmax=st['vh'] / n_slots, viol=(st['viol'] / n_slots).tolist(), hmax=hmax_list.tolist(),
               s2_rate=st['n_s2'] / max(st['n_data'], 1), diff_rate=st['n_diff'] / max(st['n_s2'], 1),
               succ_rate=st['n_succ'] / max(st['n_data'], 1), peak=st['peak'],
               compute_rate=st['compute'] / max(st['n_data'], 1),
               planner_mix=(st['n_j'] / max(st['n_data'], 1)).tolist())
    if log_multi:
        out['mlog'] = mlog
    if record_timing:
        out['timing'] = {k: [float(np.median(v_)) if v_ else None, float(np.percentile(v_, 95)) if v_ else None, len(v_)]
                         for k, v_ in tim.items()}
    if record_v:
        out['vlog'] = vlog
    if s2_delay > 0:
        out['delay'] = dict(applied=n_applied, stale=n_stale, pending_left=len(pending))
    out['Qmax'] = float(qmax) if gate in ('vq', 'multi') else None
    out['Qend'] = float(Q) if gate in ('vq', 'multi') else None
    if log_calib:
        out['calib'] = calib
    if record_trace:
        out['trace'] = {k: np.asarray(v)[::50].tolist() for k, v in trace.items()}
    return out
