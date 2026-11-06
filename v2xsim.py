"""
v2xsim.py -- Physics-grounded mmWave V2I simulator shared by both projects.

Components
----------
* Mobility   : stochastic Intelligent Driver Model (IDM) [Treiber et al., PRE 2000]
               on a 4-lane, two-direction highway with 3GPP TR 37.885-style
               wrap-around, plus randomly triggered lane changes with a
               smooth (raised-cosine) lateral profile.
* Path loss  : 3GPP TR 38.901 UMi-Street-Canyon LOS model.
* Shadowing  : log-normal, spatially correlated along each vehicle's track
               (Gudmundson exponential model).
* Blockage   : vehicle bodies act as knife edges; loss from ITU-R P.526
               single knife-edge approximation J(nu); dominant edge taken.
* Array      : RSU half-wavelength ULA (axis parallel to the road) with
               continuous phase steering; exact normalised array factor.
* Fading     : wideband Rician (K = 9 dB, 38.901 UMi LOS mean) on F
               sub-bands when unobstructed, Rayleigh when J > 3 dB.
* Sensing    : a beam-sweep slot that returns noisy (x, y) for every
               vehicle detected; position noise follows a CRB-like
               1/sqrt(SNR) law; also returns a noisy excess-loss reading.
* World model: per-vehicle constant-velocity Kalman filters (longitudinal
               and lateral) whose process noise is learned by maximum
               likelihood, combined with a deterministic physics renderer
               (geometry -> angle, path loss, knife-edge blockage) and a
               Gauss-Hermite success-probability predictor.

All randomness flows from numpy Generators seeded by the caller.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

C0 = 299_792_458.0


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # radio
    fc: float = 28e9
    B: float = 400e6
    slot: float = 1e-3               # scheduling slot [s]
    P_tx_dBm: float = 20.0
    NF_dB: float = 9.0
    G_v_dBi: float = 5.0             # vehicle antenna gain
    N_ant: int = 256                 # RSU ULA size (1.37 m aperture at 28 GHz)
    r_req: float = 4.0               # required wideband spectral eff. [b/s/Hz]
    F_sub: int = 16                  # sub-bands for wideband fading
    K_rice_dB: float = 9.0
    sigma_sf_dB: float = 4.0
    d_corr_sf: float = 10.0
    # geometry
    L_road: float = 400.0            # wrap-around length [m]
    lane_y: tuple = (1.75, 5.25, 8.75, 12.25)
    lane_dir: tuple = (+1, +1, -1, -1)
    y_rsu: float = -4.0
    h_rsu: float = 6.0
    # vehicles
    n_veh: int = 12
    truck_frac: float = 0.25
    car_len: float = 4.5
    truck_len: float = 12.0
    veh_width: float = 2.5
    car_h: float = 1.6
    truck_h: float = 3.8
    car_ant_h: float = 1.5
    truck_ant_h: float = 3.6
    # IDM
    v0_car: tuple = (25.0, 33.0)
    v0_truck: tuple = (22.0, 25.0)
    a_car: float = 1.5
    a_truck: float = 0.8
    b_comf: float = 2.0
    s0: float = 2.0
    T_head: float = 1.2
    sigma_drv: float = 0.4           # driver OU acceleration noise std [m/s^2]
    tau_drv: float = 1.0
    lc_rate: float = 0.05            # lane-change initiations per vehicle per second
    lc_dur: float = 3.0
    # sensing (beam sweep)
    sigma_pos0: float = 0.15         # position noise at reference SNR [m]
    snr_ref_dB: float = 20.0
    snr_det_dB: float = -3.0
    sigma_snr_meas_dB: float = 1.0
    warmup_s: float = 10.0
    look_slots: int = 4              # duration of one look (hierarchical 256-beam sweep)

    @property
    def lam(self):
        return C0 / self.fc

    @property
    def noise_dBm(self):
        return -174.0 + 10 * np.log10(self.B) + self.NF_dB


def wrap(x, L):
    """Map to [-L/2, L/2)."""
    return (x + L / 2) % L - L / 2


# ----------------------------------------------------------------------------
# Array, propagation, blockage
# ----------------------------------------------------------------------------
def array_gain(dx, N):
    """Normalised ULA power gain g(dx) in [0,1], dx = u - u_hat (direction-cosine
    error), half-wavelength spacing: g = |1/N sum_n exp(j pi n dx)|^2."""
    x = np.pi * np.asarray(dx, dtype=float)
    num = np.sin(N * x / 2.0)
    den = N * np.sin(x / 2.0)
    out = np.ones_like(x)
    nz = np.abs(den) > 1e-12
    out[nz] = (num[nz] / den[nz]) ** 2
    return out


def pathloss_umi_los(d3d, fc):
    """3GPP TR 38.901 UMi-Street Canyon LOS (d_2D < d'_BP regime) [dB]."""
    return 32.4 + 21.0 * np.log10(np.maximum(d3d, 1.0)) + 20.0 * np.log10(fc / 1e9)


def knife_edge_J(nu):
    """ITU-R P.526 approximation of single knife-edge diffraction loss [dB]."""
    nu = np.asarray(nu, dtype=float)
    out = np.zeros_like(nu)
    m = nu > -0.78
    v = nu[m]
    out[m] = 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)
    return out


def blockage_loss(xr, y, h_ant, body_h, body_len, cfg: Config):
    """Dominant knife-edge loss [dB] at every receiver k caused by every other
    vehicle j.  Each blocker body offers two diffraction paths: over its roof
    (vertical edge clearance) and around its nearer side (horizontal edge
    clearance); the wave takes the easier one, so the loss of blocker j is
    min(J(nu_top), J(nu_side)).  This keeps the loss continuous when the ray
    grazes a vehicle corner.  The dominant blocker is retained (single-edge
    approximation, ITU-R P.526).  Vectorised over leading batch dims (..., K)."""
    lam = cfg.lam
    yr, hr = cfg.y_rsu, cfg.h_rsu
    xk = xr[..., :, None]; yk = y[..., :, None]; hk = h_ant[..., :, None]
    xj = xr[..., None, :]; yj = y[..., None, :]
    Hj = np.broadcast_to(body_h, xr.shape)[..., None, :]
    Lj = np.broadcast_to(body_len, xr.shape)[..., None, :]
    dy = yk - yr
    f = (yj - yr) / dy                                  # fraction along path
    valid = (f > 0.0) & (f < 1.0)
    xstar = f * xk
    spread = np.abs(xk) * cfg.veh_width / np.maximum(dy, 1e-6) / 2.0
    diff = wrap(xstar - xj, cfg.L_road)
    d2d = np.sqrt(xk ** 2 + dy ** 2)
    d1 = np.maximum(f * d2d, 0.5); d2 = np.maximum((1 - f) * d2d, 0.5)
    fres = np.sqrt(2.0 * (d1 + d2) / (lam * d1 * d2))
    # roof edge
    h_los = hr + (hk - hr) * f
    nu_top = (Hj - h_los) * fres
    # side edge: horizontal penetration measured perpendicular to the ray
    sin_phi = dy / d2d
    pen = (Lj / 2.0 + spread - np.abs(diff)) * sin_phi
    nu_side = pen * fres
    J = np.minimum(knife_edge_J(nu_top), knife_edge_J(nu_side))
    K = xr.shape[-1]
    eye = np.eye(K, dtype=bool)
    J = np.where(valid & ~eye, J, 0.0)
    return J.max(axis=-1)


def geometry(xr, y, h_ant, cfg: Config):
    dy = y - cfg.y_rsu
    dh = h_ant - cfg.h_rsu
    d3 = np.sqrt(xr ** 2 + dy ** 2 + dh ** 2)
    u = xr / d3
    return d3, u


def mean_snr_dB(d3, J, sf, cfg: Config):
    """Mean SNR assuming perfect beam alignment (array gain N included)."""
    return (cfg.P_tx_dBm + 10 * np.log10(cfg.N_ant) + cfg.G_v_dBi
            - pathloss_umi_los(d3, cfg.fc) - J - sf - cfg.noise_dBm)


# ----------------------------------------------------------------------------
# Wideband fading success tables
# ----------------------------------------------------------------------------
class SuccessTable:
    """P(wideband rate >= r_req | mean SNR) for Rician (K) and Rayleigh,
    tabulated by Monte Carlo once (common across runs, fixed seed)."""
    def __init__(self, cfg: Config, n_mc=40000, seed=12345):
        rng = np.random.default_rng(seed)
        self.grid = np.arange(-15.0, 45.0001, 0.1)
        F = cfg.F_sub
        K = 10 ** (cfg.K_rice_dB / 10)
        los = np.sqrt(K / (K + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (n_mc, 1)))
        h_r = los + np.sqrt(1 / (K + 1) / 2) * (rng.standard_normal((n_mc, F)) + 1j * rng.standard_normal((n_mc, F)))
        h_n = np.sqrt(1 / 2) * (rng.standard_normal((n_mc, F)) + 1j * rng.standard_normal((n_mc, F)))
        g_r = np.abs(h_r) ** 2; g_n = np.abs(h_n) ** 2
        pr, pn = [], []
        for s in self.grid:
            gam = 10 ** (s / 10)
            pr.append(np.mean(np.mean(np.log2(1 + gam * g_r), 1) >= cfg.r_req))
            pn.append(np.mean(np.mean(np.log2(1 + gam * g_n), 1) >= cfg.r_req))
        self.p_rice = np.maximum.accumulate(np.array(pr))
        self.p_rayl = np.maximum.accumulate(np.array(pn))

    def snr_at(self, p, rayleigh=False):
        """Mean SNR [dB] at which the success probability equals p."""
        tab = self.p_rayl if rayleigh else self.p_rice
        return float(np.interp(p, tab + 1e-9 * np.arange(len(tab)), self.grid))

    def __call__(self, snr_dB, rayleigh_mask):
        pr = np.interp(snr_dB, self.grid, self.p_rice)
        pn = np.interp(snr_dB, self.grid, self.p_rayl)
        return np.where(rayleigh_mask, pn, pr)


# ----------------------------------------------------------------------------
# Environment (ground truth)
# ----------------------------------------------------------------------------
class Highway:
    def __init__(self, cfg: Config, rng: np.random.Generator, rng_ch: np.random.Generator = None):
        self.cfg = cfg
        self.rng = rng                                  # mobility + shadowing (policy independent)
        self.rng_ch = rng_ch if rng_ch is not None else np.random.default_rng(rng.integers(2 ** 62))
        K = cfg.n_veh
        nl = len(cfg.lane_y)
        self.is_truck = np.zeros(K, bool)
        nt = int(round(cfg.truck_frac * K))
        self.is_truck[rng.choice(K, nt, replace=False)] = True
        self.lane = np.arange(K) % nl
        rng.shuffle(self.lane)
        self.length = np.where(self.is_truck, cfg.truck_len, cfg.car_len)
        self.body_h = np.where(self.is_truck, cfg.truck_h, cfg.car_h)
        self.ant_h = np.where(self.is_truck, cfg.truck_ant_h, cfg.car_ant_h)
        self.v0 = np.where(self.is_truck, rng.uniform(*cfg.v0_truck, K), rng.uniform(*cfg.v0_car, K))
        self.amax = np.where(self.is_truck, cfg.a_truck, cfg.a_car)
        # longitudinal coordinate along own direction
        self.s = np.zeros(K)
        for l in range(nl):
            idx = np.where(self.lane == l)[0]
            if len(idx):
                base = np.linspace(0, cfg.L_road, len(idx), endpoint=False) + rng.uniform(0, cfg.L_road / max(len(idx), 1) * 0.5)
                self.s[idx] = base % cfg.L_road
        self.v = 0.8 * self.v0
        self.eta = np.zeros(K)                       # OU driver noise
        self.y = np.array(cfg.lane_y)[self.lane].astype(float)
        self.lc_t = np.full(K, -1.0)                 # lane-change elapsed time (-1: none)
        self.lc_from = self.y.copy(); self.lc_to = self.y.copy()
        self.lc_target = self.lane.copy()
        self.sf = rng.standard_normal(K) * cfg.sigma_sf_dB
        self.t = 0.0
        n_w = int(cfg.warmup_s / cfg.slot)
        for _ in range(n_w):
            self.step()

    # direction of travel per vehicle
    @property
    def dirn(self):
        return np.array(self.cfg.lane_dir)[self.lane]

    @property
    def x_global(self):
        L = self.cfg.L_road
        return np.where(self.dirn > 0, self.s, (L - self.s)) % L

    @property
    def xr(self):
        return wrap(self.x_global, self.cfg.L_road)

    @property
    def vx(self):
        return self.dirn * self.v

    @property
    def vy(self):
        cfg = self.cfg
        out = np.zeros(len(self.v))
        m = self.lc_t >= 0
        tau = self.lc_t[m] / cfg.lc_dur
        out[m] = (self.lc_to[m] - self.lc_from[m]) * 0.5 * np.pi / cfg.lc_dur * np.sin(np.pi * tau)
        return out

    def _leaders(self):
        """For each vehicle, gap and leader speed in its current (target) lane."""
        cfg = self.cfg
        K = len(self.s)
        gap = np.full(K, 1e3); vl = self.v.copy()
        occ = self.lc_target
        for i in range(K):
            same = np.where((occ == occ[i]) | (self.lane == occ[i]))[0]
            same = same[same != i]
            if len(same) == 0:
                continue
            ds = (self.s[same] - self.s[i]) % cfg.L_road
            j = same[np.argmin(ds)]
            gap[i] = max(ds.min() - self.length[j], 0.1)
            vl[i] = self.v[j]
        return gap, vl

    def step(self):
        cfg = self.cfg; dt = cfg.slot; rng = self.rng
        gap, vl = self._leaders()
        dv = self.v - vl
        s_star = cfg.s0 + np.maximum(0.0, self.v * cfg.T_head + self.v * dv / (2 * np.sqrt(self.amax * cfg.b_comf)))
        acc = self.amax * (1 - (self.v / self.v0) ** 4 - (s_star / gap) ** 2)
        # OU driver noise
        self.eta += -self.eta / cfg.tau_drv * dt + cfg.sigma_drv * np.sqrt(2 * dt / cfg.tau_drv) * rng.standard_normal(len(self.v))
        acc = np.clip(acc + self.eta, -8.0, 3.0)
        self.v = np.maximum(self.v + acc * dt, 0.0)
        ds = self.v * dt
        self.s = (self.s + ds) % cfg.L_road
        # shadowing (Gudmundson)
        rho = np.exp(-ds / cfg.d_corr_sf)
        self.sf = rho * self.sf + np.sqrt(1 - rho ** 2) * cfg.sigma_sf_dB * rng.standard_normal(len(self.v))
        # lane changes
        active = self.lc_t >= 0
        self.lc_t[active] += dt
        done = active & (self.lc_t >= cfg.lc_dur)
        if done.any():
            self.lane[done] = self.lc_target[done]
            self.y[done] = self.lc_to[done]
            self.lc_t[done] = -1.0
        act = self.lc_t >= 0
        tau = np.clip(self.lc_t[act] / cfg.lc_dur, 0, 1)
        self.y[act] = self.lc_from[act] + (self.lc_to[act] - self.lc_from[act]) * 0.5 * (1 - np.cos(np.pi * tau))
        start = (~act) & (rng.random(len(self.v)) < cfg.lc_rate * dt)
        for i in np.where(start)[0]:
            l = self.lane[i]
            cands = [m for m in (l - 1, l + 1) if 0 <= m < len(cfg.lane_y) and cfg.lane_dir[m] == cfg.lane_dir[l]]
            if not cands:
                continue
            m = cands[rng.integers(len(cands))]
            # gap acceptance in target lane
            oth = np.where(self.lane == m)[0]
            ok = True
            for j in oth:
                d = wrap(self.s[j] - self.s[i], cfg.L_road)
                if -(self.length[i] + 10) < d < self.length[j] + 10:
                    ok = False; break
            if ok:
                self.lc_t[i] = 0.0; self.lc_from[i] = self.y[i]
                self.lc_to[i] = cfg.lane_y[m]; self.lc_target[i] = m
        self.t += dt

    # ---------------- radio ----------------
    def true_channel(self):
        cfg = self.cfg
        xr = self.xr
        d3, u = geometry(xr, self.y, self.ant_h, cfg)
        J = blockage_loss(xr, self.y, self.ant_h, self.body_h, self.length, cfg)
        snr0 = mean_snr_dB(d3, J, self.sf, cfg)
        return dict(u=u, d3=d3, J=J, snr0=snr0)

    def transmit(self, k, u_hat):
        """Data transmission to vehicle k with steering direction u_hat.
        Returns (success, info)."""
        cfg = self.cfg; rng = self.rng_ch
        ch = self.true_channel()
        g = array_gain(np.array([ch['u'][k] - u_hat]), cfg.N_ant)[0]
        snr = 10 ** ((ch['snr0'][k] + 10 * np.log10(max(g, 1e-12))) / 10)
        F = cfg.F_sub
        if ch['J'][k] > 3.0:
            h = np.sqrt(0.5) * (rng.standard_normal(F) + 1j * rng.standard_normal(F))
        else:
            Kf = 10 ** (cfg.K_rice_dB / 10)
            h = np.sqrt(Kf / (Kf + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi)) + \
                np.sqrt(1 / (Kf + 1) / 2) * (rng.standard_normal(F) + 1j * rng.standard_normal(F))
        rate = np.mean(np.log2(1 + snr * np.abs(h) ** 2))
        return bool(rate >= cfg.r_req), dict(gain=g, u_err=ch['u'][k] - u_hat, snr0=ch['snr0'][k], J=ch['J'][k])

    def sense(self):
        """Beam-sweep sensing slot.  Returns measurements for detected vehicles:
        z_x, z_y (positions), snr_meas (dB, aligned), detected mask, sigma."""
        cfg = self.cfg; rng = self.rng_ch
        ch = self.true_channel()
        # sweep with N-beam DFT codebook in u: best codeword quantisation error
        cb = np.linspace(-1 + 1 / cfg.N_ant, 1 - 1 / cfg.N_ant, cfg.N_ant)
        qerr = np.min(np.abs(ch['u'][:, None] - cb[None, :]), axis=1)
        g = array_gain(qerr, cfg.N_ant)
        snr_sweep = ch['snr0'] + 10 * np.log10(g)
        det = snr_sweep >= cfg.snr_det_dB
        sig = cfg.sigma_pos0 * 10 ** ((cfg.snr_ref_dB - snr_sweep) / 20)
        sig = np.clip(sig, cfg.sigma_pos0, 5.0)
        zx = self.x_global + sig * rng.standard_normal(len(sig))
        zy = self.y + sig * rng.standard_normal(len(sig))
        snr_meas = ch['snr0'] + cfg.sigma_snr_meas_dB * rng.standard_normal(len(sig))
        return dict(zx=zx % cfg.L_road, zy=zy, snr=snr_meas, det=det, sigma=sig)


# ----------------------------------------------------------------------------
# World model: Kalman bank + physics renderer + success predictor
# ----------------------------------------------------------------------------
class WorldModel:
    """Physics-structured world model.  Latent per vehicle: (x, vx, y, vy) with
    constant-velocity dynamics (discretised white-noise acceleration) and
    learned process-noise intensities q_x, q_y.  Static attributes (class,
    height, length) are assumed known from sensing (e.g., radar cross-section
    classification)."""
    def __init__(self, cfg: Config, env_static: dict, q_x=0.5, q_y=0.05, mode='cv'):
        self.cfg = cfg
        self.mode = mode                      # 'cv' (imagine) or 'zoh' (hold last)
        K = cfg.n_veh
        self.q_x, self.q_y = q_x, q_y
        self.ant_h = env_static['ant_h']; self.body_h = env_static['body_h']
        self.length = env_static['length']
        self.m = np.zeros((K, 4))             # x, vx, y, vy
        self.Px = np.tile(np.diag([100.0, 100.0]), (K, 1, 1))
        self.Py = np.tile(np.diag([4.0, 1.0]), (K, 1, 1))
        self.sf_hat = np.zeros(K)
        self.sf_age_dist = np.zeros(K)
        self.age = np.full(K, 10 ** 6)        # belief age (slots since last measurement)
        self.init = np.zeros(K, bool)
        self.censored = np.zeros(K, bool)
        self.nack_hold = False

    @staticmethod
    def _FQ(dt, q):
        # continuous white-noise acceleration (CWNA) model: discretisation
        # consistent, q in m^2/s^3 (Bar-Shalom et al., Sec. 6.2)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        return F, Q

    def predict(self, n=1):
        cfg = self.cfg; dt = cfg.slot * n
        if self.mode == 'cv':
            for _ in range(n):
                F1, Qx1 = self._FQ(cfg.slot, self.q_x); _, Qy1 = self._FQ(cfg.slot, self.q_y)
                self.m[:, 0] += self.m[:, 1] * cfg.slot
                self.m[:, 2] += self.m[:, 3] * cfg.slot
                self.Px = F1 @ self.Px @ F1.T + Qx1
                self.Py = F1 @ self.Py @ F1.T + Qy1
            self.m[:, 0] %= cfg.L_road
            self.sf_age_dist += np.abs(self.m[:, 1]) * dt
        self.age = self.age + n

    def update(self, meas):
        cfg = self.cfg
        det = meas['det']
        for k in np.where(det)[0]:
            R = meas['sigma'][k] ** 2
            if self.mode == 'zoh':
                # zero-order hold: belief = last measurement, no motion model
                self.m[k] = [meas['zx'][k], 0.0, meas['zy'][k], 0.0]
                self.Px[k] = np.diag([R, 0.0]); self.Py[k] = np.diag([R, 0.0])
                self.init[k] = True; self.age[k] = 0
                continue
            if not self.init[k]:
                self.m[k] = [meas['zx'][k], 0.0, meas['zy'][k], 0.0]
                self.Px[k] = np.diag([R, 30.0 ** 2]); self.Py[k] = np.diag([R, 1.0])
                self.init[k] = True
            else:
                for (i0, P) in ((0, self.Px), (2, self.Py)):
                    z = meas['zx'][k] if i0 == 0 else meas['zy'][k]
                    innov = z - self.m[k, i0]
                    if i0 == 0:
                        innov = wrap(innov, cfg.L_road)
                    S = P[k][0, 0] + R
                    Kg = P[k][:, 0] / S
                    self.m[k, i0:i0 + 2] += Kg * innov
                    P[k] = P[k] - np.outer(Kg, P[k][0, :])
            self.m[k, 0] %= cfg.L_road
            self.age[k] = 0
        # shadowing estimate from SNR residual vs rendered geometry (incl. predicted blockage)
        ren = self.render()
        resid = ren['snr_nosf'] - meas['snr']
        self.sf_hat[det] = resid[det]
        self.sf_age_dist[det] = 0.0
        # censored observation: a missed detection implies sweep SNR < threshold,
        # i.e. excess loss at least snr_nosf - snr_det (Tobit-style lower bound)
        # Without a cause model for the excess loss (a co-moving blocker can
        # persist far beyond the shadowing decorrelation distance), the
        # censored state is held until the next look.
        miss = (~det) & self.init
        self.censored[det] = False
        if miss.any():
            lb = ren['snr_nosf'] - cfg.snr_det_dB
            self.sf_hat[miss] = np.maximum(self.sf_hat[miss], lb[miss])
            self.sf_age_dist[miss] = 0.0
            self.censored[miss] = True

    def nack(self, k, snr_cap_dB):
        """Censored link observation from a failed transmission: the mean SNR
        is believed to be at most snr_cap_dB until the next look."""
        ren = self.render()
        lb = ren['snr_nosf'][k] - snr_cap_dB
        if self.nack_hold:
            self.sf_hat[k] = max(self.sf_hat[k], lb)
            self.censored[k] = True
        else:
            # soft censoring: the excess-loss estimate is raised to the NACK
            # bound and then decorrelates with travelled distance (shadowing law)
            mu, _ = self.sf_pred()
            self.sf_hat[k] = max(mu[k], lb)
            self.sf_age_dist[k] = 0.0

    def render(self, m=None):
        """Physics renderer: belief mean -> (u, d3, J, SNR without shadowing)."""
        cfg = self.cfg
        m = self.m if m is None else m
        xr = wrap(m[..., 0], cfg.L_road); y = m[..., 2]
        d3, u = geometry(xr, y, self.ant_h, cfg)
        J = blockage_loss(xr, y, self.ant_h, self.body_h, self.length, cfg)
        snr_nosf = mean_snr_dB(d3, J, 0.0, cfg)
        return dict(u=u, d3=d3, J=J, snr_nosf=snr_nosf, xr=xr, y=y)

    def sigma_u(self, ren=None):
        """Std of the direction-cosine error from the Gaussian belief (delta method)."""
        cfg = self.cfg
        ren = self.render() if ren is None else ren
        xr, y = ren['xr'], ren['y']
        dy = y - cfg.y_rsu; dh = self.ant_h - cfg.h_rsu
        d = np.sqrt(xr ** 2 + dy ** 2 + dh ** 2)
        Jx = (d ** 2 - xr ** 2) / d ** 3
        Jy = -xr * dy / d ** 3
        var = Jx ** 2 * self.Px[:, 0, 0] + Jy ** 2 * self.Py[:, 0, 0]
        return np.sqrt(np.maximum(var, 0.0))

    def sf_pred(self, extra_dist=0.0):
        rho = np.exp(-(self.sf_age_dist + extra_dist) / self.cfg.d_corr_sf)
        rho = np.where(self.censored, 1.0, rho)
        mu = rho * self.sf_hat
        sd = self.cfg.sigma_sf_dB * np.sqrt(np.maximum(1 - rho ** 2, 0.0))
        sd = np.where(self.censored, 1.0, sd)
        return mu, sd


# Gauss-Hermite nodes for E over N(0,1)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / _GH_W.sum()


def predict_success(snr_nosf, J, sf_mu, sf_sd, u_sd, table: SuccessTable, N, u_err_samples=None):
    """Predicted P(success) per vehicle when steering at the belief mean.
    Integrates beam-error (Gaussian with std u_sd, or empirical samples) and
    residual shadowing (Gaussian) against the wideband fading table."""
    snr_nosf = np.atleast_1d(snr_nosf)
    out = np.zeros_like(snr_nosf, dtype=float)
    ray = J > 3.0
    sf_nodes = sf_mu[:, None] + sf_sd[:, None] * _GH_X[None, :]          # (K, n)
    if u_err_samples is None:
        ue = u_sd[:, None] * _GH_X[None, :]                               # (K, n)
        g_db = 10 * np.log10(np.maximum(array_gain(ue, N), 1e-12))
        wu = _GH_W[None, :]
    else:
        ue = np.broadcast_to(u_err_samples, (len(snr_nosf),) + np.shape(u_err_samples)[-1:])
        g_db = 10 * np.log10(np.maximum(array_gain(ue, N), 1e-12))
        wu = np.full((1, g_db.shape[1]), 1.0 / g_db.shape[1])
    snr = snr_nosf[:, None, None] - sf_nodes[:, :, None] + g_db[:, None, :]
    p = table(snr, ray[:, None, None])
    out = np.sum(p * _GH_W[None, :, None] * wu[:, None, :], axis=(1, 2))
    return out


def learn_process_noise(cfg: Config, seed=999, T_obs_s=30.0, every=20, grid_x=None, grid_y=None):
    """System identification of the world model: maximum (innovation) likelihood
    estimate of q_x, q_y from a training trajectory observed through the
    sensing model every `every` slots (no access to ground truth)."""
    rng = np.random.default_rng(seed)
    env = Highway(cfg, rng)
    n = int(T_obs_s / cfg.slot)
    Z = []
    for t in range(n):
        env.step()
        if t % every == 0:
            Z.append(env.sense())
    grid_x = np.geomspace(0.01, 50.0, 30) if grid_x is None else grid_x
    grid_y = np.geomspace(0.0005, 10.0, 30) if grid_y is None else grid_y
    dt = cfg.slot * every

    def nll(q, key, wrapq):
        F = np.array([[1, dt], [0, 1.0]]); Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        tot = 0.0
        for k in range(cfg.n_veh):
            m = None; P = None
            for z in Z:
                if not z['det'][k]:
                    if m is not None:
                        m = F @ m; P = F @ P @ F.T + Q
                    continue
                R = z['sigma'][k] ** 2; zz = z[key][k]
                if m is None:
                    m = np.array([zz, 0.0]); P = np.diag([R, 900.0 if key == 'zx' else 1.0]); first = 2
                    continue
                m = F @ m; P = F @ P @ F.T + Q
                inn = zz - m[0]
                if wrapq:
                    inn = wrap(inn, cfg.L_road)
                S = P[0, 0] + R
                if first > 0:
                    first -= 1
                else:
                    tot += 0.5 * (np.log(2 * np.pi * S) + inn ** 2 / S)
                Kg = P[:, 0] / S
                m = m + Kg * inn; P = P - np.outer(Kg, P[0, :])
        return tot
    lx = [nll(q, 'zx', True) for q in grid_x]
    ly = [nll(q, 'zy', False) for q in grid_y]
    return float(grid_x[int(np.argmin(lx))]), float(grid_y[int(np.argmin(ly))])
