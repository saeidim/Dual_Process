"""Numbers, table rows and figures of the second revision, from results_rev2/*.json
(and the first-revision results it compares with).  Writes PAPER/numbers2.tex and
PAPER/figs/fig_compute.pdf.  Same statistics as make_figures_rev.py: per-realization
change relative to always-S1 on the same traffic, mean and 95% Student-t interval,
paired two-sided t-tests; equivalence by two one-sided tests (TOST)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(HERE, 'results_rev'); R2 = os.path.join(HERE, 'results_rev2')
PAPER = os.environ.get('PAPER', os.path.join(HERE, '..', '..', 'paper'))
FIG = os.path.join(PAPER, 'figs'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.2, 'xtick.labelsize': 6.8,
                     'ytick.labelsize': 6.8, 'lines.linewidth': 1.0, 'lines.markersize': 3.2,
                     'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.grid': True, 'grid.alpha': 0.3})
N = {}
EQ_MARGIN = 10.0                     # equivalence margin for TOST [percentage points]


def L(d, n):
    p = os.path.join(d, n + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def arr(v, k):
    return np.array([r[k] for r in v], float)


def ci(x):
    x = np.asarray(x, float); n = len(x)
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def rel(x, ref):
    return 100 * (np.asarray(x) - np.asarray(ref)) / np.asarray(ref)


def num(x, d=1):
    s = f'{x:.{d}f}'
    return '0.0' if float(s) == 0 else s


def fmt_p(p):
    """p-value as a math-mode expression including 'p' (use as $\\MACRO$ in the text)"""
    return 'p<0.001' if p < 0.001 else f'p={p:.3f}'


def paired(a, b):
    d = np.asarray(a) - np.asarray(b); m, h = ci(d)
    return m, h, float(stats.ttest_rel(a, b).pvalue)


def tost(a, b, margin=EQ_MARGIN):
    """p-value of the two one-sided tests that |mean(a-b)| < margin."""
    d = np.asarray(a) - np.asarray(b); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((d.mean() + margin) / se, n - 1)
    p2 = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(p1, p2))


def put(tag, x, ref, key='viol_hmax', sign=-1):
    """store the reduction (sign=-1) of a metric relative to ref, with its CI"""
    m, h = ci(rel(arr(x, key), arr(ref, key)))
    N[tag] = num(sign * m); N[tag + 'CI'] = f'{h:.1f}'
    N[tag + 'CHG'] = ('+' if m >= 0 else '') + num(m)          # signed change, for math mode
    return m, h


# ----------------------------- references ---------------------------------
main = L(R1, 'main'); s1 = main['s1']; s2 = main['s2']
s1v = arr(s1, 'viol_hmax'); s1c = arr(s1, 'cost')
dv = lambda x: rel(arr(x, 'viol_hmax'), s1v[:len(x)])
dc = lambda x: rel(arr(x, 'cost'), s1c[:len(x)])
cal1 = L(R1, 'calib'); tune1 = L(R1, 'tune')['chosen']
gmax = float(np.max(cal1['vals'][1])); V1 = float(tune1['vq'])
N['MIXTERM'] = f"{2 * (2 * V1 * gmax + 1) / V1:.0f}"
v1 = np.array(cal1['v']); ghat1 = np.maximum(np.interp(v1, cal1['knots'][1], cal1['vals'][1]), 0)
N['OPTCAL'] = f"{np.mean(ghat1 * (v1 >= cal1['thr']['0.1']['static'])):.2f}"
N['QBNDMAINR'] = f"{V1 * gmax + 1 - 0.1:.1f}"; N['QMAXBNDR'] = f"{50 * gmax + 1 - 0.1:.0f}"

# ----------------------------- cheap planners ------------------------------
ch = L(R2, 'cheap'); dl = L(R2, 'delay'); tg = L(R2, 'tiny_gates'); ml = L(R2, 'multi_low')
cs = L(R2, 'cs1'); ln = L(R2, 'long'); lr = L(R2, 'learned'); mm2 = L(R2, 'mismatch2'); t2 = L(R2, 'tune2')
pts = []                                   # (compute per slot, violation change, CI, label, style)
if ch is not None:
    for k, tag in [('s2_tiny', 'TINY'), ('s2_m2h16', 'MTWOHSIXTEEN'), ('s2_lite', 'LITE')]:
        put(f'CHV{tag}', ch[k], s1); put(f'CHC{tag}', ch[k], s1, 'cost')
    full_v = dv(s2); tiny_v = dv(ch['s2_tiny']); lite_v = dv(ch['s2_lite'])
    m, h, p = paired(tiny_v, full_v); N['DTINYFULL'] = num(m); N['DTINYFULLCI'] = f'{h:.1f}'; N['PTINYFULL'] = fmt_p(p)
    N['TOSTTINYFULL'] = fmt_p(tost(tiny_v, full_v)); N['EQMARGIN'] = f'{EQ_MARGIN:g}'
    m, h, p = paired(dc(ch['s2_tiny']), dc(s2)); N['DTINYFULLC'] = num(m); N['DTINYFULLCCI'] = f'{h:.1f}'; N['PTINYFULLC'] = fmt_p(p)
    N['TOSTTINYFULLC'] = fmt_p(tost(dc(ch['s2_tiny']), dc(s2)))
    tt = np.array([r['timing']['s2'][0] for r in ch['s2_tiny']]) * 1e3
    N['TTINY'] = f'{np.median(tt):.1f}'
    N['NCHEAP'] = str(len(ch['s2_tiny']))
    pts += [(1.0, ci(full_v), 'always full', 'full'), (0.125, ci(lite_v), 'always light', 'lite'),
            (1 / 16, ci(dv(ch['s2_m2h16'])), r'always $M$=2, $H$=16', 'mid'), (1 / 32, ci(tiny_v), 'always tiny', 'tiny')]
if dl is not None:
    meta = dl['_meta']; d_t = meta['d_tiny']
    N['TFULLHERE'] = f"{meta['t_full_here_ms']:.0f}"; N['TRATIOTINY'] = f"{1 / meta['ratio']:.0f}"; N['DTINY'] = str(d_t)
    if ch is not None:
        N['TTINYSCALED'] = f"{40.0 * meta['ratio']:.1f}"
        N['TINYSPEEDUP'] = f"{40.0 * meta['ratio']:.0f}"
    for k, tag in [('s2_d40', 'SDFORTY'), ('aq_0.2_d40', 'AQDFORTY'), (f's2tiny_d{d_t}', 'TINYD'),
                   (f's2tiny_d{d_t}_ahead', 'TINYDAHEAD'), (f'aqtiny_0.2_d{d_t}', 'AQTINYD')]:
        put(f'DLV{tag}', dl[k], s1); put(f'DLC{tag}', dl[k], s1, 'cost')
        ap = np.array([r['delay']['applied'] for r in dl[k]], float); stl = np.array([r['delay']['stale'] for r in dl[k]], float)
        N[f'DLSTALE{tag}'] = f'{100 * np.sum(stl) / max(np.sum(ap + stl), 1):.0f}'
    N['DLCTINYDAHEADNEG'] = num(-float(N['DLCTINYDAHEAD']))
    sw = L(R2, 'delay_sweep')
    if sw is not None and ch is not None:
        SW = {}
        for d_, lt in [(1, 'A'), (2, 'B'), (4, 'C')]:
            mv_, hv_ = ci(dv(sw[f's2tiny_d{d_}'])); N[f'SWV{lt}'] = num(-mv_); N[f'SWV{lt}CI'] = f'{hv_:.1f}'
            SW[d_] = (N[f'SWV{lt}'], N[f'SWV{lt}CI'])
        N['DELAYSWEEPTEXT'] = ('A latency sweep with the tiny planner locates the loss: with plans applied $d=1$, 2 and 4 data slots '
                               'later, the violation reduction is $' + SW[1][0] + '\\pm' + SW[1][1] + '$\\%, $' + SW[2][0] + '\\pm' + SW[2][1]
                               + '$\\% and $' + SW[4][0] + '\\pm' + SW[4][1] + '$\\%, against $' + N['CHVTINY'] + '\\pm' + N['CHVTINYCI'] + '$\\% without delay.')
    else:
        N['DELAYSWEEPTEXT'] = ''
    m, h, p = paired(dv(dl['s2_d40']), dv(s2)); N['DDFORTY'] = num(m); N['DDFORTYCI'] = f'{h:.1f}'; N['PDFORTY'] = fmt_p(p)
    if ch is not None:
        m, h, p = paired(dv(dl[f's2tiny_d{d_t}']), dv(ch['s2_tiny'])); N['DDTINY'] = num(m); N['DDTINYCI'] = f'{h:.1f}'; N['PDTINY'] = fmt_p(p)
if tg is not None and ch is not None:
    for rho in [0.1, 0.2]:
        rt = {0.1: 'TEN', 0.2: 'TWENTY'}[rho]
        for g in ['random', 'static', 'aq', 'vq']:
            k = f'{g}_{rho}'
            put(f'TG{g.upper()}{rt}', tg[k], s1); put(f'TGC{g.upper()}{rt}', tg[k], s1, 'cost')
            N[f'TGRATE{g.upper()}{rt}'] = f"{arr(tg[k], 's2_rate').mean():.3f}"
            N[f'TGMAX{g.upper()}{rt}'] = f"{arr(tg[k], 's2_rate').max():.3f}"
            N[f'TGSHARE{g.upper()}{rt}'] = f"{100 * dv(tg[k]).mean() / tiny_v.mean():.0f}"
        for g in ['static', 'aq', 'vq']:
            m, h, p = paired(dv(tg[f'{g}_{rho}']), dv(tg[f'random_{rho}']))
            N[f'TGD{g.upper()}{rt}'] = num(m); N[f'TGD{g.upper()}{rt}CI'] = f'{h:.1f}'; N[f'TGP{g.upper()}{rt}'] = fmt_p(p)
        m, h, p = paired(dv(tg[f'aq_{rho}']), tiny_v); N[f'TGDAQTINY{rt}'] = num(m); N[f'TGDAQTINY{rt}CI'] = f'{h:.1f}'; N[f'TGPAQTINY{rt}'] = fmt_p(p)
        for g, lab in [('aq', 'tiny, VQ-signal'), ('vq', 'tiny, VQ-gain'), ('random', 'tiny, random')]:
            pts.append((arr(tg[f'{g}_{rho}'], 's2_rate').mean() / 32, ci(dv(tg[f'{g}_{rho}'])), f'{lab}', 'tg' + g))
if ml is not None:
    for k in sorted(ml):
        rb = float(k.split('_')[1]); tag = {0.00313: 'A', 0.00625: 'B'}[round(rb, 5)]
        put(f'MLV{tag}', ml[k], s1); put(f'MLC{tag}', ml[k], s1, 'cost')
        N[f'MLRATE{tag}'] = f"{arr(ml[k], 'compute_rate').mean():.4f}"; N[f'MLBUD{tag}'] = f'{rb:.4f}'
        mix = np.mean([r['planner_mix'] for r in ml[k]], 0)
        N[f'MLMIXTINY{tag}'] = f'{100 * mix[1]:.1f}'; N[f'MLMIXLITE{tag}'] = f'{100 * mix[2]:.1f}'; N[f'MLMIXFULL{tag}'] = f'{100 * mix[3]:.1f}'
        pts.append((arr(ml[k], 'compute_rate').mean(), ci(dv(ml[k])), 'menu (tiny, light, full)', 'menu'))
# full-planner gates at 0.1 and 0.2 (first revision) for the compute frontier
fr = L(R1, 'frontier'); r02 = L(R1, 'rho02')
if r02 is not None and fr is not None:
    aq2 = sorted(fr['aq_0.2'] + r02['aq_0.2'], key=lambda r: r['seed']); rn2 = sorted(fr['random_0.2'] + r02['random_0.2'], key=lambda r: r['seed'])
    pts += [(arr(main['aq_0.1'], 's2_rate').mean(), ci(dv(main['aq_0.1'])), 'full, VQ-signal', 'fullaq'),
            (arr(aq2, 's2_rate').mean(), ci(dv(aq2)), 'full, VQ-signal', 'fullaq'),
            (arr(main['vq_0.1'], 's2_rate').mean(), ci(dv(main['vq_0.1'])), 'full, VQ-gain', 'fullvq'),
            (arr(main['random_0.1'], 's2_rate').mean(), ci(dv(main['random_0.1'])), 'full, random', 'fullrnd'),
            (arr(rn2, 's2_rate').mean(), ci(dv(rn2)), 'full, random', 'fullrnd')]
if pts:
    sty = dict(full=('s', '0.35'), lite=('s', '0.55'), mid=('s', '0.7'), tiny=('s', 'k'),
               tgaq=('D', '#17becf'), tgvq=('v', '#2ca02c'), tgrandom=('o', '#ff7f0e'), menu=('*', '#d62728'),
               fullaq=('D', '#1f77b4'), fullvq=('v', '#556b2f'), fullrnd=('o', '#9467bd'))
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    seen = set()
    groups = {}
    for c_, (m_, h_), lab, s_ in pts:
        groups.setdefault(s_, []).append((c_, m_, h_, lab))
    for s_, g in groups.items():
        g.sort(); mk, col = sty[s_]
        x = [a[0] for a in g]; y = [a[1] for a in g]; e = [a[2] for a in g]
        ax.errorbar(x, y, yerr=e, fmt='-' + mk, color=col, capsize=1.5, lw=0.9, elinewidth=0.6,
                    label=g[0][3], ms=4 if mk == '*' else 3.2)
    ax.set_xscale('log'); ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel('compute per data slot [full-planner units]'); ax.set_ylabel('change in violations vs. S1 [%]')
    ax.legend(frameon=False, fontsize=5.4, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.24),
              handlelength=1.6, columnspacing=0.8)
    fig.savefig(os.path.join(FIG, 'fig_compute.pdf'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ----------------------------- cost-aware System 1 -------------------------
if cs is not None and t2 is not None:
    N['SONEDELTA'] = f"{t2['chosen']['delta']:g}"
    fe = lambda x: (f'{x:.0e}').replace('e-0', 'e-').replace('1e-', '10^{-') + '}' if x < 0.01 else f'{x:g}'
    N['TUNEETATINYA'] = fe(t2['chosen']['aqT0.1']); N['TUNEETATINYB'] = fe(t2['chosen']['aqT0.2'])
    N['TUNEVTINYA'] = f"{t2['chosen']['vqT0.1']:g}"; N['TUNEVTINYB'] = f"{t2['chosen']['vqT0.2']:g}"
    s1ca = cs['s1ca']
    put('CAVSONE', s1ca, s1); put('CACSONE', s1ca, s1, 'cost')
    m, h, p = paired(arr(s1ca, 'viol_hmax') * 100, s1v * 100); N['CAPVSONE'] = fmt_p(p)
    N['CASONEVIOL'] = f"{100 * arr(s1ca, 'viol_hmax').mean():.2f}"
    for k, tag in [('s2ca', 'STWO'), ('s2ca_tiny', 'TINY'), ('aqca_0.2', 'AQ'), ('aqca_tiny_0.2', 'AQTINY')]:
        put(f'CAV{tag}', cs[k], s1ca); put(f'CAC{tag}', cs[k], s1ca, 'cost')
        N[f'CARATE{tag}'] = f"{arr(cs[k], 's2_rate').mean():.3f}"
    put('CAVSTWOVSONE', cs['s2ca'], s1)
    N['CASHARE'] = f"{100 * np.mean(rel(arr(s1ca, 'viol_hmax'), s1v)) / np.mean(dv(s2)):.0f}"
    N['CAAQMAXRATE'] = f"{arr(cs['aqca_tiny_0.2'], 's2_rate').max():.3f}"
    m, h, p = paired(arr(cs['s2ca'], 'viol_hmax'), arr(s2, 'viol_hmax')); N['CAPSTWOVSSTWO'] = fmt_p(p)
    N['CASTWOVIOL'] = f"{100 * arr(cs['s2ca'], 'viol_hmax').mean():.2f}"; N['STWOVIOL'] = f"{100 * arr(s2, 'viol_hmax').mean():.2f}"
    for d_ in [0, 4, 8, 16]:
        N[f'CATUNE{ {0: "ZERO", 4: "FOUR", 8: "EIGHT", 16: "SIXTEEN"}[d_] }'] = f"{t2['table'][f's1ca|{d_}']['cost']:.1f}"

# ----------------------------- learned gate --------------------------------
if lr is not None:
    meta = lr['_meta']
    N['SPLEARN'] = f"{meta['spearman_learned']:.2f}"; N['SPV'] = f"{meta['spearman_v']:.2f}"
    for k, tag in [('vq_learned', 'VQ'), ('pd_learned', 'PD')]:
        put(f'LRV{tag}', lr[k], s1); put(f'LRC{tag}', lr[k], s1, 'cost'); N[f'LRRATE{tag}'] = f"{arr(lr[k], 's2_rate').mean():.3f}"
    N['LRSHAREVQ'] = f"{100 * dv(lr['vq_learned']).mean() / dv(s2).mean():.0f}"
    m, h, p = paired(dv(lr['vq_learned']), dv(main['vq_0.1'])); N['LRDVQ'] = num(m); N['LRDVQCI'] = f'{h:.1f}'; N['LRPVQ'] = fmt_p(p)
    m, h, p = paired(dv(lr['vq_learned']), dv(main['random_0.1'])); N['LRDRND'] = num(m); N['LRDRNDCI'] = f'{h:.1f}'; N['LRPRND'] = fmt_p(p)

# ----------------------------- world-model errors --------------------------
d6 = L(R1, 'design'); ed2 = L(R2, 'eps_diag2')
if mm2 is not None and d6 is not None:
    rows = []
    for k, tag, lab in [('pl-12', 'PLMTW', 'path loss $-12$~dB (optimistic)'), ('pl-6', 'PLMSIX', 'path loss $-6$~dB'),
                        ('pl+6', 'PLPSIX', 'path loss $+6$~dB'), ('pl+12', 'PLPTW', 'path loss $+12$~dB (pessimistic)'),
                        ('block3', 'BLTHREE', 'blockage loss $\\times3$')]:
        a_v = rel(arr(mm2[k], 'viol_hmax'), arr(d6['S1'], 'viol_hmax')); a_c = rel(arr(mm2[k], 'cost'), arr(d6['S1'], 'cost'))
        mv, hv = ci(a_v); mc, hc = ci(a_c)
        N[f'MMV{tag}'] = num(-mv); N[f'MMV{tag}CI'] = f'{hv:.1f}'; N[f'MMV{tag}NEG'] = num(mv); N[f'MMC{tag}'] = num(-mc); N[f'MMC{tag}CI'] = f'{hc:.1f}'
        dp = '--' if ed2 is None else f"{ed2[k]['mean']:.4f} / {ed2[k]['p99']:.2f}"
        sg = lambda x: (f'{x:+.1f}').replace('-', '$-$')
        rows.append(f"{lab} & {dp} & {sg(mc)} $\\pm$ {hc:.1f} & {sg(mv)} $\\pm$ {hv:.1f}\\\\")
        if ed2 is not None:
            N[f'DPV{tag}'] = f"{ed2[k]['mean']:.3f}"; N[f'DPVMAX{tag}'] = f"{ed2[k]['p99']:.2f}"
    N['MISMROWSTWO'] = '\n'.join(rows)

# ----------------------------- long traces ---------------------------------
if ln is not None:
    s1l = ln['s1']; N['NLONGSEEDS'] = str(len(s1l)); N['NLONGSLOTS'] = str(24000)
    for k, tag in [('s2', 'STWO'), ('s2_tiny', 'TINY'), ('aq_0.2', 'AQ'), ('random_0.2', 'RND')]:
        a_v = rel(arr(ln[k], 'viol_hmax'), arr(s1l, 'viol_hmax'))
        N[f'LGV{tag}'] = num(-a_v.mean()); N[f'LGV{tag}MIN'] = num(-a_v.max()); N[f'LGV{tag}MAX'] = num(-a_v.min())
    N['LGRATEAQ'] = f"{arr(ln['aq_0.2'], 's2_rate').mean():.3f}"
    N['LGSHAREAQ'] = f"{100 * np.mean(rel(arr(ln['aq_0.2'], 'viol_hmax'), arr(s1l, 'viol_hmax'))) / np.mean(rel(arr(ln['s2'], 'viol_hmax'), arr(s1l, 'viol_hmax'))):.0f}"

with open(os.path.join(PAPER, 'numbers2.tex'), 'w') as f:
    f.write('% generated by make_numbers_rev2.py (second revision); do not edit\n')
    for k, val in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(json.dumps({k: v for k, v in N.items() if 'ROWS' not in k}, indent=0))
