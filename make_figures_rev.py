"""Builds every figure, table row and number of the revised manuscript from
results_rev/*.json.  Writes ../../paper/figs/*.pdf and ../../paper/numbers.tex
(paths can be changed with the PAPER environment variable).

Statistics: for every realization the change of a metric relative to always-S1
on the same traffic is 100 (x - x_S1) / x_S1; we report its mean and the 95%
Student-t interval over realizations.  Gates are compared with each other by a
paired t-test on the per-realization changes (two-sided), with a Wilcoxon
signed-rank test as a robustness check and Holm's correction over the baselines."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results_rev')
PAPER = os.environ.get('PAPER', os.path.join(HERE, '..', '..', 'paper'))
FIG = os.path.join(PAPER, 'figs'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'xtick.labelsize': 6.8,
                     'ytick.labelsize': 6.8, 'lines.linewidth': 1.0, 'lines.markersize': 3,
                     'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.grid': True, 'grid.alpha': 0.3})
SZ = (2.0, 1.7)
COL = dict(s1='k', s2='0.45', random='#ff7f0e', periodic='#8c564b', unc='#9467bd', static='#1f77b4',
           vq='#2ca02c', aq='#17becf', pd='#d62728', oracle='#e377c2')
LAB = dict(s1='always-S1', s2='always-S2', random='random', periodic='periodic', unc='uncertainty',
           static='static threshold', vq='VQ, gain scale (prop.)', aq='VQ, signal scale (prop.)', pd='primal-dual')
N = {}


def L(name):
    p = os.path.join(RES, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def arr(v, key):
    return np.array([r[key] for r in v], float)


def rel(x, ref):
    n = min(len(x), len(ref))
    return 100 * (np.asarray(x[:n]) - np.asarray(ref[:n])) / np.asarray(ref[:n])


def pchg(x, ref):
    return ci(rel(x, ref))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fmt_p(p):
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def sgn(x, d=1):
    s = f'{x:+.{d}f}'
    return s.replace('-', '$-$') if not s.startswith('+0.0') and s not in ('-0.0',) else s.replace('-0.0', '0.0').replace('+0.0', '0.0')


def num(x, d=1):
    """number for use inside math mode in the text (plain minus becomes a math minus)"""
    s = f'{x:.{d}f}'
    return '0.0' if float(s) == 0 else s


# =========================== calibration ===========================
cal = L('calib')
g_all = np.array(cal['g']); gf = g_all[:, 1]; v = np.array(cal['v']); unc = np.array(cal['unc'])
gself = np.array(cal['gself'])[:, 1]
N['NCAL'] = str(len(v))
N['GZERO'] = f"{100*np.mean(gf == 0):.0f}"
N['GNEG'] = f"{100*np.mean(gf < 0):.1f}"
N['SELFBIAS'] = f"{100*(gself.mean()-gf.mean())/gf.mean():.0f}"
o = np.argsort(-gf)
N['TOPTEN'] = f"{100*gf[o[:len(gf)//10]].sum()/gf.sum():.0f}"
gmax = float(np.max(cal['vals'][1]))
N['GMAX'] = f"{gmax:.1f}"


def capture(score, rhos, g=gf):
    oo = np.argsort(-score); cs = np.cumsum(g[oo]) / g.sum()
    return np.array([cs[max(int(r * len(g)) - 1, 0)] for r in rhos])


fig, ax = plt.subplots(figsize=SZ)
gs = np.sort(gf[gf > 0])
ccdf = (len(gs) - np.arange(len(gs))) / len(gf)
ax.loglog(gs, ccdf, color=COL['vq'])
ax.set_xlabel('deliberation gain $G_t$ (reference-measured)'); ax.set_ylabel('$\\mathbb{P}(G_t>x)$')
save(fig, 'fig_gain_ccdf.pdf')
rh = np.linspace(0.005, 0.5, 100)
fig, ax = plt.subplots(figsize=SZ)
ax.plot(rh, capture(gf, rh), color=COL['oracle'], label='measured gain (oracle)')
ax.plot(rh, capture(v, rh), color=COL['vq'], label='risk signal $v_t$')
ax.plot(rh, capture(unc, rh), color=COL['unc'], label='belief uncertainty')
ax.plot(rh, rh * 1.0, ':', color='k', label='random')
ax.set_xlabel('deliberation budget $\\rho$'); ax.set_ylabel('captured gain fraction'); ax.set_xlim(0, 0.5)
ax.set_ylim(-0.1, 1.6)
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper center', ncol=2, fontsize=5.8,
          bbox_to_anchor=(0.5, 1.02), handlelength=1.5, columnspacing=0.8)
save(fig, 'fig_capture.pdf')
N['CAPRISK'] = f"{100*capture(v, [0.1])[0]:.0f}"; N['CAPORACLE'] = f"{100*capture(gf, [0.1])[0]:.0f}"
N['CAPUNC'] = f"{100*capture(unc, [0.1])[0]:.0f}"
N['NEGLITE'] = f"{100*np.mean(g_all[:, 0] < 0):.1f}"; N['NEGFULL'] = f"{100*np.mean(g_all[:, 1] < 0):.1f}"
N['BHIGHV'] = f"{100*np.mean(v >= 0.8):.0f}"; N['BTOPTEN'] = N['TOPTEN']; N['BCAPTEN'] = N['CAPRISK']

# =========================== main comparison at rho = 0.1 ===========================
mr = L('main')
GATES = ['random', 'periodic', 'unc', 'static', 'aq', 'pd', 'vq']
s1 = mr['s1']; s1c = arr(s1, 'cost'); s1v = arr(s1, 'viol_hmax'); s1a = arr(s1, 'aoi')
N['NSEEDS'] = str(len(s1))
N['VIOLSHARE'] = f"{100*np.mean([r['cost'] - r['aoi'] for r in s1]) / np.mean(s1c):.0f}"
N['SONEVIOL'] = f"{100*s1v.mean():.2f}"
EF = {}
DAG = {'static': '$^\\dagger$'}
for gname in ['s2'] + GATES:
    key = 's2' if gname == 's2' else f'{gname}_0.1'
    vv = mr[key]
    EF[gname] = dict(c=rel(arr(vv, 'cost'), s1c), v=rel(arr(vv, 'viol_hmax'), s1v), a=rel(arr(vv, 'aoi'), s1a),
                     rate=1.0 if gname == 's2' else arr(vv, 's2_rate').mean(), raw=vv)
rows = [f"{LAB['s1']} & {s1c.mean():.1f} & 0 (ref.) & {s1a.mean():.2f} & {100*s1v.mean():.2f} & 0 (ref.) & 0\\\\"]
for gname in GATES + ['s2']:
    e = EF[gname]; vv = e['raw']
    mc, hc = ci(e['c']); mv, hv = ci(e['v'])
    rows.append(f"{LAB[gname]}{DAG.get(gname, '')} & {arr(vv, 'cost').mean():.1f} & {sgn(mc)} $\\pm$ {hc:.1f} & {arr(vv, 'aoi').mean():.2f} & "
                f"{100*arr(vv, 'viol_hmax').mean():.2f} & {sgn(mv)} $\\pm$ {hv:.1f} & {e['rate']:.3f}\\\\")
N['MAINROWS'] = '\n'.join(rows)
for gname, tag in [('s2', 'ALL'), ('vq', 'VQ'), ('static', 'STAT'), ('random', 'RND'), ('periodic', 'PER'),
                   ('unc', 'UNC'), ('aq', 'AQ'), ('pd', 'PD')]:
    mv, hv = ci(EF[gname]['v']); mc, hc = ci(EF[gname]['c']); ma, ha = ci(EF[gname]['a'])
    N[f'VIOL{tag}'] = num(-mv); N[f'VIOL{tag}CI'] = f"{hv:.1f}"
    N[f'COST{tag}'] = num(-mc); N[f'COST{tag}CI'] = f"{hc:.1f}"
    N[f'AOI{tag}'] = num(ma); N[f'AOI{tag}CI'] = f"{ha:.1f}"
    N[f'RATE{tag}'] = f"{EF[gname]['rate']:.3f}"
N['STATRATEPC'] = f"{100*EF['static']['rate']:.1f}"
# share of the always-S2 reduction obtained by the virtual queue (paired bootstrap interval)
rng = np.random.default_rng(0)
n = len(s1)
def share(idx, key):
    return np.mean(EF['vq'][key][idx]) / np.mean(EF['s2'][key][idx])
for key, tag in [('v', 'VIOL'), ('c', 'COST')]:
    bs = [share(rng.integers(0, n, n), key) for _ in range(5000)]
    N[f'CAPTURE{tag}'] = f"{100*share(np.arange(n), key):.0f}"
    N[f'CAPTURE{tag}LO'] = f"{100*np.percentile(bs, 2.5):.0f}"; N[f'CAPTURE{tag}HI'] = f"{100*np.percentile(bs, 97.5):.0f}"

# paired tests: virtual queue against each other gate
tests = []
for gname in ['random', 'periodic', 'unc', 'static', 'aq', 'pd', 's2']:
    for key in ['v', 'c']:
        d = EF['vq'][key] - EF[gname][key]
        m, h = ci(d)
        p_t = stats.ttest_rel(EF['vq'][key], EF[gname][key]).pvalue
        p_w = stats.wilcoxon(EF['vq'][key], EF[gname][key]).pvalue
        tests.append(dict(g=gname, k=key, m=m, h=h, pt=p_t, pw=p_w))
# Holm over the six baselines, separately for each metric
for key in ['v', 'c']:
    tt = [t for t in tests if t['k'] == key and t['g'] != 's2']
    order = np.argsort([t['pt'] for t in tt]); m_ = len(tt); prev = 0.0
    for r_, i in enumerate(order):
        adj = min(1.0, max(prev, (m_ - r_) * tt[i]['pt'])); tt[i]['holm'] = adj; prev = adj
    for t in tests:
        if t['k'] == key and t['g'] == 's2':
            t['holm'] = t['pt']
trows = []
for gname in ['random', 'periodic', 'unc', 'static', 'aq', 'pd', 's2']:
    tv = [t for t in tests if t['g'] == gname and t['k'] == 'v'][0]
    tc = [t for t in tests if t['g'] == gname and t['k'] == 'c'][0]
    trows.append(f"{LAB[gname]} & {sgn(tv['m'])} $\\pm$ {tv['h']:.1f} & {fmt_p(tv['holm'])} & {fmt_p(tv['pw'])} & "
                 f"{sgn(tc['m'])} $\\pm$ {tc['h']:.1f} & {fmt_p(tc['holm'])}\\\\")
    for t in (tv, tc):
        tag = {'random': 'RND', 'periodic': 'PER', 'unc': 'UNC', 'static': 'STAT', 'aq': 'AQ', 'pd': 'PD', 's2': 'ALL'}[gname]
        N[f'D{tag}{t["k"].upper()}'] = num(t['m']); N[f'D{tag}{t["k"].upper()}CI'] = f"{t['h']:.1f}"
        N[f'P{tag}{t["k"].upper()}'] = fmt_p(t['holm'])
N['TESTROWS'] = '\n'.join(trows)
# other budget-enforcing gates against the non-selective baselines (unadjusted paired t-tests)
for g1, t1 in [('aq', 'AQ'), ('pd', 'PD'), ('vq', 'VQ')]:
    for g2, t2 in [('random', 'RND'), ('periodic', 'PER'), ('unc', 'UNC')]:
        d = EF[g1]['v'] - EF[g2]['v']; m_, h_ = ci(d)
        N[f'X{t1}{t2}'] = num(m_); N[f'X{t1}{t2}CI'] = f"{h_:.1f}"
        N[f'XP{t1}{t2}'] = fmt_p(stats.ttest_rel(EF[g1]['v'], EF[g2]['v']).pvalue)
# untuned step sizes (first run, V = 10 and eta0 = 1): budget overspent
for k, tag in [('vq_untuned_0.1', 'VQTEN'), ('pd_untuned_0.1', 'PDONE'), ('aq_untuned_0.1', 'AQFAST')]:
    if k in mr:
        vv = mr[k]; mv, hv = ci(rel(arr(vv, 'viol_hmax'), s1v)); mc, hc = ci(rel(arr(vv, 'cost'), s1c))
        N[f'VIOL{tag}'] = num(-mv); N[f'VIOL{tag}CI'] = f"{hv:.1f}"; N[f'COST{tag}'] = num(-mc)
        N[f'RATE{tag}'] = f"{arr(vv, 's2_rate').mean():.3f}"; N[f'RATEMAX{tag}'] = f"{arr(vv, 's2_rate').max():.3f}"
N['RATEMAXSTAT'] = f"{arr(mr['static_0.1'], 's2_rate').max():.3f}"; N['RATEMINSTAT'] = f"{arr(mr['static_0.1'], 's2_rate').min():.3f}"
N['RATEMAXVQ'] = f"{arr(mr['vq_0.1'], 's2_rate').max():.3f}"
tun = L('tune')
if tun is not None:
    ch = tun['chosen']; N['TUNEV'] = f"{ch['vq']:g}"; N['TUNEETA'] = f"{ch['aq']:g}"; N['TUNEETAZ'] = f"{ch['pd']:g}"; N['TUNEVM'] = f"{ch['multi']:g}"
# per-realization heterogeneity: share of always-S2 reduction obtained by VQ
srate = arr(mr['static_0.1'], 's2_rate'); lo = srate <= 0.1
N['NLOWDENS'] = str(int(lo.sum()))
N['SHARELOW'] = f"{100*EF['vq']['v'][lo].mean()/EF['s2']['v'][lo].mean():.0f}"
N['SHAREHIGH'] = f"{100*EF['vq']['v'][~lo].mean()/EF['s2']['v'][~lo].mean():.0f}"
bd = L('burst_diag')
if bd is not None:
    m = bd['mean']
    N['HIGHFRAC'] = f"{100*m['s1']['highfrac']:.1f}"
    for k, tag in [('vq_0.1', 'VQ'), ('aq_0.1', 'AQ'), ('pd_0.1', 'PD'), ('random_0.1', 'RND')]:
        N[f'LOWDELIB{tag}'] = f"{100*m[k]['frac_low']:.0f}"; N[f'COVER{tag}'] = f"{100*m[k]['cover']:.0f}"


# timing
tm1 = np.array([r['timing']['s1'][0] for r in mr['s1']]) * 1e3
tm1p = np.array([r['timing']['s1'][1] for r in mr['s1']]) * 1e3
tm2 = np.array([r['timing']['s2'][0] for r in mr['s2']]) * 1e3
tm2p = np.array([r['timing']['s2'][1] for r in mr['s2']]) * 1e3
N['TSONE'] = f"{np.median(tm1):.2f}"; N['TSONEP'] = f"{np.median(tm1p):.1f}"
N['TSTWO'] = f"{np.median(tm2):.0f}"; N['TSTWOP'] = f"{np.median(tm2p):.0f}"
N['TRATIO'] = f"{np.median(tm2)/np.median(tm1):.0f}"
N['COMPUTEX'] = f"{np.mean(arr(mr['s2'], 'wall'))/np.mean(arr(mr['s1'], 'wall')):.0f}"

# =========================== frontier ===========================
fr = L('frontier')
if fr is not None:
    nF = len(fr['random_0.05']); N['NFRONT'] = str(nF)
    rhos = [0.05, 0.1, 0.2, 0.3]
    figs = {k: plt.subplots(figsize=SZ) for k in ['viol', 'cost']}
    for gname in ['random', 'static', 'pd', 'vq', 'aq']:
        rate, dv, dc = [], [], []
        for r in rhos:
            vv = mr[f'{gname}_0.1'][:nF] if r == 0.1 else fr[f'{gname}_{r}']
            rate.append(arr(vv, 's2_rate').mean())
            dv.append(ci(rel(arr(vv, 'viol_hmax'), s1v[:nF]))[0]); dc.append(ci(rel(arr(vv, 'cost'), s1c[:nF]))[0])
        mk = 'D' if gname in ('vq', 'aq') else 'o'
        figs['viol'][1].plot(rate, dv, '-' + mk, color=COL[gname], label=LAB[gname])
        figs['cost'][1].plot(rate, dc, '-' + mk, color=COL[gname], label=LAB[gname])
        if gname == 'static':
            N['STATLOWRATE'] = f"{100*rate[0]:.1f}"; N['STATLOWVIOL'] = num(-dv[0])
    for k, key, ref in [('viol', 'viol_hmax', s1v), ('cost', 'cost', s1c)]:
        ax = figs[k][1]
        ax.axhline(0.0, color=COL['s1'], ls='--', lw=0.8, label=LAB['s1'])
        ax.axhline(ci(rel(arr(mr['s2'][:nF], key), ref[:nF]))[0], color=COL['s2'], ls='-.', lw=0.8, label=LAB['s2'])
        ax.set_xlabel('realized deliberation rate')
    figs['viol'][1].set_ylabel('change in violations vs. S1 [%]'); figs['cost'][1].set_ylabel('change in cost vs. S1 [%]')
    figs['viol'][0].legend(*figs['viol'][1].get_legend_handles_labels(), frameon=False, loc='lower center', ncol=2, fontsize=5.6,
                           bbox_to_anchor=(0.55, 0.98), columnspacing=0.8, handlelength=1.6)
    save(figs['viol'][0], 'fig_front_viol.pdf'); save(figs['cost'][0], 'fig_front_cost.pdf')

# rho = 0.2 on all realizations (frontier seeds + rho02 stage)
r2 = L('rho02')
if fr is not None and r2 is not None:
    E2 = {}
    for gname in ['random', 'vq', 'aq']:
        vv = fr[f'{gname}_0.2'] + r2[f'{gname}_0.2']
        vv.sort(key=lambda r: r['seed'])
        E2[gname] = dict(v=rel(arr(vv, 'viol_hmax'), s1v), c=rel(arr(vv, 'cost'), s1c), rate=arr(vv, 's2_rate'))
        tag = {'random': 'RND', 'vq': 'VQ', 'aq': 'AQ'}[gname]
        m_, h_ = ci(E2[gname]['v']); N[f'TWO{tag}V'] = num(-m_); N[f'TWO{tag}VCI'] = f"{h_:.1f}"
        m_, h_ = ci(E2[gname]['c']); N[f'TWO{tag}C'] = num(-m_); N[f'TWO{tag}CCI'] = f"{h_:.1f}"
        N[f'TWO{tag}R'] = f"{E2[gname]['rate'].mean():.3f}"; N[f'TWO{tag}RMAX'] = f"{E2[gname]['rate'].max():.3f}"
    N['NTWO'] = str(len(E2['aq']['v']))
    for a_, b_, tag in [('aq', 'random', 'AQRND'), ('aq', 'vq', 'AQVQ')]:
        d = E2[a_]['v'] - E2[b_]['v']; m_, h_ = ci(d)
        N[f'TWOD{tag}'] = num(m_); N[f'TWOD{tag}CI'] = f"{h_:.1f}"; N[f'TWOP{tag}'] = fmt_p(stats.ttest_rel(E2[a_]['v'], E2[b_]['v']).pvalue)
    d = E2['aq']['v'] - EF['s2']['v']; m_, h_ = ci(d)
    N['TWODAQSTWO'] = num(m_); N['TWODAQSTWOCI'] = f"{h_:.1f}"; N['TWOPAQSTWO'] = fmt_p(stats.ttest_rel(E2['aq']['v'], EF['s2']['v']).pvalue)
    N['SHARETWOAQ'] = f"{100*E2['aq']['v'].mean()/EF['s2']['v'].mean():.0f}"
if fr is not None:
    for r in [0.05, 0.2, 0.3]:
        for gname in ['random', 'static', 'vq', 'aq', 'pd']:
            m_, h_ = pchg(arr(fr[f'{gname}_{r}'], 'viol_hmax'), s1v)
            rt_ = {0.05: 'FIVE', 0.2: 'TWENTY', 0.3: 'THIRTY'}[r]
            N[f'FR{rt_}{gname.upper()}'] = num(-m_); N[f'FR{rt_}{gname.upper()}CI'] = f"{h_:.1f}"
    N['FRSTWO'] = num(-ci(rel(arr(mr['s2'], 'viol_hmax'), s1v)[:nF])[0]); N['FRSTWOCI'] = f"{ci(rel(arr(mr['s2'], 'viol_hmax'), s1v)[:nF])[1]:.1f}"

# =========================== V sweep ===========================
vs = L('vsweep')
if vs is not None:
    lam = cal['thr']['0.1']['lam']
    N['COLDRATE'] = f"{arr(vs['vq_V10.0_cold'], 's2_rate').mean():.3f}"
    vsw = {k: x for k, x in vs.items() if not k.endswith('_cold')}
    Vs = sorted(float(k.split('V')[1]) for k in vsw)
    fig, ax = plt.subplots(figsize=SZ)
    for V_, ls in zip(Vs, ['-', '--', '-.', ':', '-']):
        tr = vsw[f'vq_V{V_}'][0]['trace']; z = np.array(tr['z_avg']); n_ = (np.arange(len(z)) * 50 + 1)
        ax.semilogx(n_, z, ls, label=f'$V$={V_:g}')
    V_ = max(Vs)
    qmax_obs = max(r['Qmax'] for r in vsw[f'vq_V{V_}'])
    y0 = V_ * lam
    ax.semilogx(n_, np.minimum(0.1 + (V_ * gmax + 1 - 0.1 - y0) / n_, 1.0), color='r', lw=0.8, label='bound, $V$=%g' % V_)
    ax.axhline(0.1, color='k', lw=0.6)
    ax.set_ylim(0, 0.5); ax.set_xlabel('data slot $n$'); ax.set_ylabel('running deliberation rate')
    ax.legend(frameon=False, ncol=2, fontsize=5.8)
    save(fig, 'fig_budget_trace.pdf')
    N['QMAXOBS'] = f"{qmax_obs:.0f}"; N['QMAXBND'] = f"{V_*gmax+1-0.1:.0f}"
    fig, ax = plt.subplots(figsize=SZ); ax2 = ax.twinx()
    cm = [arr(vsw[f'vq_V{V_}'], 'cost').mean() for V_ in Vs]; rt = [arr(vsw[f'vq_V{V_}'], 's2_rate').mean() for V_ in Vs]
    ax.semilogx(Vs, cm, '-o', color=COL['vq'], label='cost'); ax2.semilogx(Vs, rt, '--s', color='0.4', label='deliberation rate')
    ax2.set_ylim(0, max(0.3, 1.1 * max(rt))); ax.set_xlabel('$V$'); ax.set_ylabel('average cost'); ax2.set_ylabel('deliberation rate')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6); ax2.grid(False)
    save(fig, 'fig_V.pdf')
    N['VCOSTLO'] = f"{cm[0]:.1f}"; N['VCOSTHI'] = f"{cm[-1]:.1f}"; N['VRATEHI'] = f"{rt[-1]:.3f}"
    N['VRATETEN'] = f"{rt[Vs.index(10.0)]:.3f}"; N['VCOSTTEN'] = f"{cm[Vs.index(10.0)]:.1f}"
    N['VRATEONE'] = f"{rt[Vs.index(1.0)]:.3f}"; N['VCOSTONE'] = f"{cm[Vs.index(1.0)]:.1f}"
    N['VRATEFOUR'] = f"{rt[Vs.index(4.0)]:.3f}"; N['VCOSTFOUR'] = f"{cm[Vs.index(4.0)]:.1f}"
    N['VCOLDCOST'] = f"{arr(vs['vq_V10.0_cold'], 'cost').mean():.1f}"
    N['VRATELO'] = f"{rt[0]:.3f}"
    N['NV'] = str(len(vsw[f'vq_V{Vs[0]}']))
# backlog in the main runs
_vt = L('tune')['chosen']['vq'] if L('tune') else 10.0
N['QMAXMAIN'] = f"{max(r['Qmax'] for r in mr['vq_0.1']):.1f}"; N['QBNDMAIN'] = f"{_vt*gmax+1-0.1:.1f}"

# =========================== design study ===========================
d6 = L('design')
if d6 is not None:
    s1c_ = arr(d6['S1'], 'cost'); s1v_ = arr(d6['S1'], 'viol_hmax')
    N['DSGNSEEDS'] = str(len(d6['S1']))
    fig, axs = None, None
    for pre, vals, fname, xl in [('M', [2, 4, 8, 16], 'fig_M.pdf', 'imagined rollouts $M$'),
                                 ('H', [8, 16, 32, 64], 'fig_H.pdf', 'planning horizon $H$')]:
        fig, ax = plt.subplots(figsize=SZ)
        for key_m, col, lab, off in [('viol_hmax', COL['vq'], 'violations', -0.04), ('cost', COL['static'], 'cost', 0.04)]:
            ms, hs = [], []
            for x in vals:
                key = f'{pre}{x}' if not (pre == 'H' and x == 32) else 'M16'
                m_, h_ = ci(rel(arr(d6[key], key_m), s1v_ if key_m == 'viol_hmax' else s1c_)); ms.append(m_); hs.append(h_)
                xs_ = {2: 'TWO', 4: 'FOUR', 8: 'EIGHT', 16: 'SIXTEEN', 32: 'THIRTYTWO', 64: 'SIXTYFOUR'}[x]
                N[f'DSG{pre}{xs_}{"V" if key_m == "viol_hmax" else "C"}'] = num(-m_)
                N[f'DSG{pre}{xs_}{"V" if key_m == "viol_hmax" else "C"}CI'] = f"{h_:.1f}"
            ax.errorbar(np.array(vals) * 2 ** off, ms, yerr=hs, fmt='-o', color=col, capsize=2, lw=1, label=lab)
        ax.axhline(0, color='k', ls=':', lw=0.8, label='always-S1')
        ax.set_xscale('log', base=2); ax.set_xticks(vals); ax.set_xticklabels([str(x) for x in vals])
        ax.minorticks_off()
        ax.set_xlabel(xl); ax.set_ylabel('change vs. S1 [%]')
        ax.legend(frameon=False, fontsize=5.8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8)
        save(fig, fname)

# =========================== multi-planner ===========================
me = L('multi')
if me is not None:
    nM = len(me['vq_multi']); N['MNSEEDS'] = str(nM)
    s1v_ = s1v[:nM]; s1c_ = s1c[:nM]
    src = dict(s2_full=mr['s2'][:nM], s2_lite=me['s2_lite'], vq_full=me['vq_full'], vq_lite=me['vq_lite'], vq_multi=me['vq_multi'])
    keys = [('s2_lite', 'always light'), ('s2_full', 'always full'), ('vq_lite', 'VQ, light only'),
            ('vq_full', 'VQ, full only'), ('vq_multi', 'VQ, multi-planner')]
    fig, ax = plt.subplots(figsize=(2.05, 1.75))
    labels = []
    for i, (k, lab) in enumerate(keys):
        m_, h_ = pchg(arr(src[k], 'viol_hmax'), s1v_)
        col = COL['vq'] if k == 'vq_multi' else ('0.6' if k.startswith('s2') else COL['static'])
        ax.barh(i, m_, xerr=h_, color=col, height=0.6, error_kw=dict(lw=0.7, capsize=1.5))
        cr = 1.0 if k == 's2_full' else (0.125 if k == 's2_lite' else arr(src[k], 'compute_rate').mean())
        labels.append(f'{lab} [{cr:.2f}]')
        tag = {'s2_lite': 'SLITE', 's2_full': 'SFULL', 'vq_lite': 'VQLITE', 'vq_full': 'VQFULL', 'vq_multi': 'VQMULTI'}[k]
        N[f'MV{tag}'] = num(-m_); N[f'MV{tag}CI'] = f"{h_:.1f}"
        mc_, hc_ = pchg(arr(src[k], 'cost'), s1c_); N[f'MCC{tag}'] = num(-mc_); N[f'MCC{tag}CI'] = f"{hc_:.1f}"
        if k.startswith('vq'):
            N[f'MC{tag}'] = f"{cr:.3f}"
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=6.2); ax.invert_yaxis()
    ax.axvline(0, color='k', lw=0.6); ax.set_xlabel('change in violations vs. S1 [%]')
    save(fig, 'fig_multi.pdf')
    mix = np.mean([r['planner_mix'] for r in me['vq_multi']], axis=0)
    N['MIXANY'] = f"{100*(1-mix[0]):.1f}"; N['MIXNONE'] = f"{100*mix[0]:.1f}"; N['MIXLITE'] = f"{100*mix[1]:.1f}"; N['MIXFULL'] = f"{100*mix[2]:.1f}"
    comp = arr(me['vq_multi'], 'compute_rate').mean()
    N['FULLSHARE'] = f"{100*mix[2]/(mix[1]*0.125+mix[2]):.0f}"
    # paired comparison multi vs light-only (violations)
    d = rel(arr(me['vq_multi'], 'viol_hmax'), s1v_) - rel(arr(me['vq_lite'], 'viol_hmax'), s1v_)
    m_, h_ = ci(d); N['MULTIMINUSLITE'] = num(m_); N['MULTIMINUSLITECI'] = f"{h_:.1f}"
    N['MULTIMINUSLITEP'] = fmt_p(stats.ttest_rel(rel(arr(me['vq_multi'], 'viol_hmax'), s1v_), rel(arr(me['vq_lite'], 'viol_hmax'), s1v_)).pvalue)
    d = rel(arr(me['vq_multi'], 'viol_hmax'), s1v_) - rel(arr(me['vq_full'], 'viol_hmax'), s1v_)
    m_, h_ = ci(d); N['MULTIMINUSFULL'] = num(m_); N['MULTIMINUSFULLCI'] = f"{h_:.1f}"
    N['MULTIMINUSFULLP'] = fmt_p(stats.ttest_rel(rel(arr(me['vq_multi'], 'viol_hmax'), s1v_), rel(arr(me['vq_full'], 'viol_hmax'), s1v_)).pvalue)
    # value maps
    fig, ax = plt.subplots(figsize=SZ)
    vv = np.linspace(0, 1, 400)
    for i, (lab, col) in enumerate([('light ($M$=4, $H$=16)', '#ff7f0e'), ('full ($M$=16, $H$=32)', COL['vq'])]):
        ax.plot(vv, np.interp(vv, cal['knots'][i], cal['vals'][i]), color=col, label=lab)
    ax.axhline(0, color='k', lw=0.6); ax.set_xlabel('risk signal $v$'); ax.set_ylabel('$\\hat g_j(v)$')
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_yticks([-1, 0, 1, 10, 100]); ax.set_yticklabels(['$-1$', '0', '1', '10', '100'])
    ax.legend(frameon=False, loc='upper left', fontsize=6)
    save(fig, 'fig_multi_maps.pdf')
    # transient explanation of compute overshoot: realized excess = (Y_n - Y_0 - projections)/n
    lm = cal['lam_multi']
    vm_ = tun['chosen']['multi'] if tun is not None else 10.0
    for k, tag, y0 in [('vq_multi', 'VQMULTI', vm_ * lm['both']), ('vq_full', 'VQFULL', vm_ * lm['full'])]:
        N[f'YEND{tag}'] = f"{np.mean([r['Qend'] for r in me[k]]):.1f}"; N[f'YZERO{tag}'] = f"{y0:.1f}"
        N[f'YMAX{tag}'] = f"{np.max([r['Qmax'] for r in me[k]]):.1f}"
    N['YBNDMULTI'] = f"{vm_*np.max(cal['vals'][0] + cal['vals'][1])/0.125 + 1:.0f}"

# =========================== dense scenario ===========================
de = L('dense'); dc = L('dense_calib')
if de is not None and dc is not None:
    s1v_ = arr(de['s1'], 'viol_hmax'); s1c_ = arr(de['s1'], 'cost')
    keys = [('random_0.1', 'random'), ('static_0.1', 'static threshold'), ('vq_0.1', 'VQ-gain'),
            ('vq_0.3', 'VQ-gain, $\\rho$=0.3'), ('s2', 'always-S2')]
    fig, ax = plt.subplots(figsize=(2.3, 1.55))
    labels = []
    for i, (k, lab) in enumerate(keys):
        m_, h_ = pchg(arr(de[k], 'viol_hmax'), s1v_)
        g_ = k.split('_')[0]
        ax.barh(i, m_, xerr=h_, color=COL.get(g_, COL['vq']) if k != 's2' else '0.6', height=0.6, error_kw=dict(lw=0.7, capsize=1.5))
        rt = 1.0 if k == 's2' else arr(de[k], 's2_rate').mean()
        labels.append(f'{lab} [{rt:.2f}]')
        tag = {'random_0.1': 'RND', 'static_0.1': 'STAT', 'vq_0.1': 'VQ', 'vq_0.3': 'VQTHIRTY', 's2': 'STWO'}[k]
        N[f'DV{tag}'] = num(-m_); N[f'DV{tag}CI'] = f"{h_:.1f}"
        mc_, hc_ = pchg(arr(de[k], 'cost'), s1c_); N[f'DC{tag}'] = num(-mc_); N[f'DC{tag}CI'] = f"{hc_:.1f}"
        N[f'DR{tag}'] = f"{rt:.3f}"
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=6.4); ax.invert_yaxis()
    ax.axvline(0, color='k', lw=0.6); ax.set_xlabel('change in violations vs. S1 [%]')
    save(fig, 'fig_dense.pdf')
    gd = np.array(dc['g'])[:, 0]; vd = np.array(dc['v'])
    od = np.argsort(-vd); cs_ = np.cumsum(gd[od]) / gd.sum()
    N['DHIGHV'] = f"{100*np.mean(vd >= 0.8):.0f}"; N['DCAPTEN'] = f"{100*cs_[int(0.1*len(vd))-1]:.0f}"
    N['DBUDSEVENTY'] = f"{100*(np.argmax(cs_ >= 0.7)+1)/len(vd):.0f}"
    N['DTOPTEN'] = f"{100*np.sort(gd)[::-1][:len(gd)//10].sum()/gd.sum():.0f}"
    cs_b = np.cumsum(gf[np.argsort(-v)]) / gf.sum(); N['BBUDSEVENTY'] = f"{100*(np.argmax(cs_b >= 0.7)+1)/len(v):.0f}"
    N['DSONEVIOL'] = f"{100*s1v_.mean():.2f}"; N['DNSEEDS'] = str(len(de['s1']))

# =========================== tolerable age ===========================
hm = L('hmax')
if hm is not None:
    ha_ = L('hmax_aq') or {}
    hm = dict(hm); hm.update(ha_)
    rows = []; nH = len(hm['s1_1.5']); N['NHMAX'] = str(nH)
    for hf in [1.5, 2.0, 3.0]:
        if hf == 2.0:
            s1_ = mr['s1'][:nH]; st_ = mr['static_0.1'][:nH]; vq_ = mr['vq_0.1'][:nH]; aq_ = mr['aq_0.1'][:nH]
        else:
            s1_ = hm[f's1_{hf}']; st_ = hm[f'static_{hf}']; vq_ = hm[f'vq_{hf}']; aq_ = hm.get(f'aq_{hf}', vq_)
        a, ha = pchg(arr(st_, 'viol_hmax'), arr(s1_, 'viol_hmax')); b, hb = pchg(arr(vq_, 'viol_hmax'), arr(s1_, 'viol_hmax'))
        c_, hc_ = pchg(arr(aq_, 'viol_hmax'), arr(s1_, 'viol_hmax'))
        rows.append(f"{hf:g}$K$ & {100*arr(s1_, 'viol_hmax').mean():.2f} & {sgn(a)} $\\pm$ {ha:.1f} & {sgn(b)} $\\pm$ {hb:.1f} & {sgn(c_)} $\\pm$ {hc_:.1f}\\\\")
        tag = {1.5: 'A', 2.0: 'B', 3.0: 'C'}[hf]
        N[f'HM{tag}STAT'] = num(-a); N[f'HM{tag}STATCI'] = f"{ha:.1f}"; N[f'HM{tag}VQ'] = num(-b); N[f'HM{tag}VQCI'] = f"{hb:.1f}"
        N[f'HM{tag}AQ'] = num(-c_); N[f'HM{tag}AQCI'] = f"{hc_:.1f}"
    N['HMAXROWS'] = '\n'.join(rows)

# =========================== world-model perturbations ===========================
mm = L('mismatch')
if mm is not None and d6 is not None:
    s1v_ = arr(d6['S1'], 'viol_hmax'); s1c_ = arr(d6['S1'], 'cost')
    rows = []
    epsd = L('eps_diag')
    for k, tag, lab in [('nom', 'NOM', 'nominal'), ('q0.01', 'LOW', 'process noise $\\times0.01$'), ('q100', 'HIGH', 'process noise $\\times100$'),
                        ('nosf', 'NOSF', 'no shadowing uncertainty'), ('noblock', 'NOBLOCK', 'no blockage')]:
        vv = d6['M16'] if k == 'nom' else mm[k]
        m_, h_ = pchg(arr(vv, 'viol_hmax'), s1v_); mc_, hc_ = pchg(arr(vv, 'cost'), s1c_)
        N[f'MM{tag}'] = num(-m_); N[f'MM{tag}CI'] = f"{h_:.1f}"; N[f'MMC{tag}'] = num(-mc_); N[f'MMC{tag}CI'] = f"{hc_:.1f}"
        dp = '--' if (epsd is None or k == 'nom') else f"{epsd[k]['mean']:.4f} / {epsd[k]['p99']:.2f}"
        rows.append(f"{lab} & {dp} & {sgn(mc_)} $\\pm$ {hc_:.1f} & {sgn(m_)} $\\pm$ {h_:.1f}\\\\")
    N['MISMROWS'] = '\n'.join(rows)
    if epsd is not None:
        for k, tag in [('q0.01', 'LOW'), ('q100', 'HIGH'), ('nosf', 'NOSF'), ('noblock', 'NOBLOCK')]:
            N[f'DP{tag}'] = f"{epsd[k]['mean']:.4f}"; N[f'DPMAX{tag}'] = f"{epsd[k]['p99']:.2f}"

with open(os.path.join(PAPER, 'numbers.tex'), 'w') as f:
    for k, val in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(json.dumps({k: val for k, val in N.items() if 'ROWS' not in k}, indent=0))
