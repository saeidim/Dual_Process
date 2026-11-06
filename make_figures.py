"""Builds every figure and every number quoted in Paper 2 from results/*.json.
Writes ../paper/figs/*.pdf and ../paper/numbers.tex."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, '..', 'paper', 'figs')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 5.5, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'lines.linewidth': 1.0, 'lines.markersize': 3,
                     'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.grid': True, 'grid.alpha': 0.3})
SZ = (2.0, 1.65)
COL = dict(s1='k', s2='0.45', random='#ff7f0e', periodic='#8c564b', unc='#9467bd', static='#1f77b4',
           vq='#2ca02c', oracle='#d62728')
LAB = dict(s1='always-S1', s2='always-S2', random='random', periodic='periodic', unc='uncertainty',
           static='static threshold', vq='virtual queue (prop.)', oracle='oracle gate')
N = {}


def L(name):
    return json.load(open(os.path.join(RES, name + '.json')))


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return x.mean(), 0.0
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def arr(v, key):
    return np.array([r[key] for r in v])


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# ---------------- calibration: gain distribution and capture ----------------
cal = L('calibration'); c = cal['cal']
g = np.array(c['g']); v = np.array(c['v']); unc = np.array(c['unc'])
gmax = float(np.max(cal['vals']))
fig, ax = plt.subplots(figsize=SZ)
gs = np.sort(g[g > 1e-12])
ccdf = 1.0 - np.arange(len(gs)) / len(g) - (np.sum(g <= 1e-12) / len(g))
ax.loglog(gs, np.maximum(ccdf, 1.0 / len(g)), color=COL['vq'])
ax.set_xlabel('deliberation gain $G_t$'); ax.set_ylabel('$\\mathbb{P}(G_t>x)$')
save(fig, 'fig_gain_ccdf.pdf')
N['GZERO'] = f"{100*np.mean(g <= 1e-12):.0f}"
o = np.argsort(-g); N['TOPTEN'] = f"{100*g[o[:len(g)//10]].sum()/g.sum():.0f}"


def capture(score, rhos):
    o = np.argsort(-score); cs = np.cumsum(g[o]) / g.sum()
    return np.array([cs[max(int(r * len(g)) - 1, 0)] for r in rhos])


rh = np.linspace(0.01, 1.0, 100)
fig, ax = plt.subplots(figsize=SZ)
ax.plot(rh, capture(g, rh), color=COL['oracle'], label='S2 gain estimates')
ax.plot(rh, capture(v, rh), color=COL['vq'], label='risk signal (17)'.replace('(17)', ''))
ax.plot(rh, capture(unc, rh), color=COL['unc'], label='belief uncertainty')
ax.plot(rh, rh, ':', color='k', label='random')
ax.set_xlabel('deliberation budget $\\rho$'); ax.set_ylabel('captured gain fraction'); ax.set_xlim(0, 0.5)
ax.legend(frameon=False, loc='lower right')
save(fig, 'fig_capture.pdf')
N['CAPRISK'] = f"{100*capture(v, [0.1])[0]:.0f}"; N['CAPORACLE'] = f"{100*capture(g, [0.1])[0]:.0f}"
N['CAPUNC'] = f"{100*capture(unc, [0.1])[0]:.0f}"

# ---------------- main frontier (against the REALIZED deliberation rate) ----------------
mr = L('main_rho')
rhos = sorted({float(k.split('_')[1]) for k in mr if '_' in k})
s1c = arr(mr['s1'], 'cost'); s1v = arr(mr['s1'], 'viol_hmax'); s1a = arr(mr['s1'], 'aoi')


def pchg(x, ref):                    # paired % change vs always-S1 on matching realizations
    n = len(x)
    return ci(100 * (x - ref[:n]) / ref[:n])


fig1, ax1 = plt.subplots(figsize=SZ); fig2, ax2 = plt.subplots(figsize=SZ)
for gname in ['random', 'periodic', 'unc', 'static', 'vq']:
    rate = [arr(mr[f'{gname}_{r}'], 's2_rate').mean() for r in rhos]
    dv = [pchg(arr(mr[f'{gname}_{r}'], 'viol_hmax'), s1v)[0] for r in rhos]
    da = [pchg(arr(mr[f'{gname}_{r}'], 'aoi'), s1a)[0] for r in rhos]
    mk = 'D' if gname == 'vq' else 'o'
    ax1.plot(rate, dv, '-' + mk, color=COL[gname], label=LAB[gname])
    ax2.plot(rate, da, '-' + mk, color=COL[gname], label=LAB[gname])
for ax, key, ref in [(ax1, 'viol_hmax', s1v), (ax2, 'aoi', s1a)]:
    ax.axhline(0.0, color=COL['s1'], ls='--', label=LAB['s1'])
    ax.axhline(pchg(arr(mr['s2'], key), ref)[0], color=COL['s2'], ls='-.', label=LAB['s2'])
    ax.set_xlabel('realized deliberation rate')
ax1.set_ylabel('change in violations vs. S1 [%]'); ax2.set_ylabel('change in mean AoI vs. S1 [%]')
ax2.legend(frameon=False, loc='upper center', ncol=2, fontsize=4.8); ax2.set_ylim(-15, 22)
save(fig1, 'fig_cost_rho.pdf'); save(fig2, 'fig_viol_rho.pdf')

VA = {}
for gname, key in [('s2', 's2'), ('static', 'static_0.1'), ('vq', 'vq_0.1'), ('random', 'random_0.1'),
                   ('periodic', 'periodic_0.1'), ('unc', 'unc_0.1')]:
    VA[gname] = (pchg(arr(mr[key], 'viol_hmax'), s1v), pchg(arr(mr[key], 'aoi'), s1a))
N['VIOLALL'] = f"{-VA['s2'][0][0]:.1f}"; N['VIOLALLCI'] = f"{VA['s2'][0][1]:.1f}"
N['AOIALL'] = f"{VA['s2'][1][0]:+.1f}"; N['AOIALLCI'] = f"{VA['s2'][1][1]:.1f}"
for gname, tag in [('vq', 'VQ'), ('static', 'STAT'), ('random', 'RND'), ('periodic', 'PER'), ('unc', 'UNC')]:
    N[f'VIOL{tag}'] = f"{-VA[gname][0][0]:.1f}"; N[f'VIOL{tag}CI'] = f"{VA[gname][0][1]:.1f}"
N['CAPTUREVQ'] = f"{100*VA['vq'][0][0]/VA['s2'][0][0]:.0f}"
N['STATRATE'] = f"{100*arr(mr['static_0.1'], 's2_rate').mean():.1f}"
N['VQRATE'] = f"{arr(mr['vq_0.1'], 's2_rate').mean():.3f}"
N['STATLOWRATE'] = f"{100*arr(mr['static_0.05'], 's2_rate').mean():.1f}"
N['STATLOWVIOL'] = f"{-pchg(arr(mr['static_0.05'], 'viol_hmax'), s1v)[0]:.1f}"
N['COSTALL'] = f"{pchg(arr(mr['s2'], 'cost'), s1c)[0]:+.1f}"; N['COSTALLCI'] = f"{pchg(arr(mr['s2'], 'cost'), s1c)[1]:.1f}"
N['COMPUTEX'] = f"{np.mean(arr(mr['s2'], 'wall'))/np.mean(arr(mr['s1'], 'wall')):.0f}"
rows = []
for gname, key in [('s1', 's1'), ('random', 'random_0.1'), ('periodic', 'periodic_0.1'), ('unc', 'unc_0.1'),
                   ('static', 'static_0.1'), ('vq', 'vq_0.1'), ('s2', 's2')]:
    vv = mr[key]
    if gname == 's1':
        rows.append(f"{LAB[gname]} & {arr(vv, 'aoi').mean():.2f} & 0 (ref.) & {100*arr(vv, 'viol_hmax').mean():.2f} & 0 (ref.) & 0\\\\")
        continue
    (dv, hv), (da, ha) = VA[gname]
    rate = 1.0 if gname == 's2' else arr(vv, 's2_rate').mean()
    rows.append(f"{LAB[gname]} & {arr(vv, 'aoi').mean():.2f} & {da:+.1f} $\\pm$ {ha:.1f} & {100*arr(vv, 'viol_hmax').mean():.2f} & {dv:+.1f} $\\pm$ {hv:.1f} & {rate:.3f}\\\\")
N['MAINROWS'] = '\n'.join(rows)
N['NSEEDS'] = str(len(mr['s1']))

# ---------------- V sweep and budget traces ----------------
vs = L('V_sweep')
N['COLDRATE'] = f"{arr(vs['vq_V50.0_cold'], 's2_rate').mean():.3f}"
vs = {k: v for k, v in vs.items() if not k.endswith('_cold')}
Vs = sorted(float(k.split('V')[1]) for k in vs)
fig, ax = plt.subplots(figsize=SZ)
qmax_obs = 0.0
for V, ls in zip(Vs, ['-', '--', '-.', ':', '-']):
    tr = vs[f'vq_V{V}'][0]['trace']; z = np.array(tr['z_avg']); n = (np.arange(len(z)) * 50 + 1)
    ax.semilogx(n, z, ls, label=f'$V$={V:g}')
    if V == max(Vs):
        qmax_obs = max(max(r['trace']['Q']) for r in vs[f'vq_V{V}'])
        q0 = V * float(np.interp(cal['thr']['0.1']['static'], cal['knots'], cal['vals']))
        ax.semilogx(n, np.minimum(0.1 + (V * gmax + 1 - q0) / n, 1.0), color='r', lw=0.8, label='bound, $V$=%g' % V)
ax.axhline(0.1, color='k', lw=0.6)
ax.set_ylim(0, 0.5); ax.set_xlabel('data slot $n$'); ax.set_ylabel('running deliberation rate')
ax.legend(frameon=False, ncol=2)
save(fig, 'fig_budget_trace.pdf')
N['QMAXOBS'] = f"{qmax_obs:.1f}"; N['QMAXBND'] = f"{max(Vs)*gmax+1:.1f}"
fig, ax = plt.subplots(figsize=SZ); ax2 = ax.twinx()
cm = [arr(vs[f'vq_V{V}'], 'cost').mean() for V in Vs]; rt = [arr(vs[f'vq_V{V}'], 's2_rate').mean() for V in Vs]
ax.semilogx(Vs, cm, '-o', color=COL['vq'], label='cost'); ax2.semilogx(Vs, rt, '--s', color='0.4', label='deliberation rate')
ax2.set_ylim(0, 0.3); ax.set_xlabel('$V$'); ax.set_ylabel('average freshness cost'); ax2.set_ylabel('deliberation rate')
ax.legend(frameon=False, loc='upper center'); ax2.legend(frameon=False, loc='center right'); ax2.grid(False)
save(fig, 'fig_V.pdf')
N['VCOSTLO'] = f"{cm[0]:.2f}"; N['VCOSTHI'] = f"{cm[-1]:.2f}"; N['VRATEHI'] = f"{rt[-1]:.3f}"
N['VWARMFIFTY'] = f"{rt[Vs.index(50.0)]:.3f}"
N['VRATETEN'] = f"{rt[Vs.index(10.0)]:.3f}"; N['VCOSTTEN'] = f"{cm[Vs.index(10.0)]:.2f}"

# ---------------- System 2 design ----------------
ds = L('design')
Ms = sorted(int(k[1:]) for k in ds if k.startswith('M'))
Hs = sorted(int(k[1:]) for k in ds if k.startswith('H') and k[1:].isdigit())
s1_half = None
fig, ax = plt.subplots(figsize=SZ)
cmM = [arr(ds[f'M{M}'], 'cost').mean() for M in Ms]
ax.semilogx(Ms, cmM, '-o', color=COL['vq'], label='always-S2'); ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
ax.set_xlabel('imagined rollouts $M$'); ax.set_ylabel('average freshness cost')
ax.axhline(arr(ds['S1'], 'cost').mean(), ls=':', color='k', label='always-S1')
ax.legend(frameon=False)
save(fig, 'fig_M.pdf')
s1d = arr(ds['S1'], 'cost').mean()
rM = {M: 100 * (c_ - s1d) / s1d for M, c_ in zip(Ms, cmM)}
N['MTWO'] = f"{rM[2]:.1f}"
N['MRANGELO'] = f"{-max(v for M, v in rM.items() if M >= 4):.1f}"; N['MRANGEHI'] = f"{-min(v for M, v in rM.items() if M >= 4):.1f}"
Hs_all = sorted(Hs + [32])
cmH = [arr(ds[f'H{H}'], 'cost').mean() if H != 32 else arr(ds['M16'], 'cost').mean() for H in Hs_all]
fig, ax = plt.subplots(figsize=SZ)
ax.semilogx(Hs_all, cmH, '-o', color=COL['vq'], label='always-S2'); ax.set_xticks(Hs_all); ax.set_xticklabels([str(h) for h in Hs_all])
ax.axhline(arr(ds['S1'], 'cost').mean(), ls=':', color='k', label='always-S1')
ax.set_xlabel('planning horizon $H$'); ax.set_ylabel('average freshness cost'); ax.legend(frameon=False)
save(fig, 'fig_H.pdf')
rH = [100 * (c_ - s1d) / s1d for c_ in cmH]
N['HRANGELO'] = f"{-max(rH):.1f}"; N['HRANGEHI'] = f"{-min(rH):.1f}"

with open(os.path.join(HERE, '..', 'paper', 'numbers.tex'), 'w') as f:
    for k, val in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(json.dumps({k: val for k, val in N.items() if 'ROWS' not in k}, indent=0))

# =====================================================================
# Journal-version expansion: multi-planner, dense scenario, design6,
# tolerable-age sensitivity, world-model mismatch
# =====================================================================
def paired(xs, ref):
    n = min(len(xs), len(ref))
    return ci(100 * (np.asarray(xs[:n]) - np.asarray(ref[:n])) / np.asarray(ref[:n]))


if os.path.exists(os.path.join(RES, 'multi_eval.json')):
    me = L('multi_eval'); mc = L('multi_calib')
    s1v_ = arr(me['s1'], 'viol_hmax'); s1a_ = arr(me['s1'], 'aoi')
    keys = [('s2_lite', 'always light'), ('s2_full', 'always full'), ('vq_lite', 'VQ, light only'),
            ('vq_full', 'VQ, full only'), ('vq_multi', 'VQ, multi-planner')]
    fig, ax = plt.subplots(figsize=SZ)
    for i, (k, lab) in enumerate(keys):
        m_, h_ = paired(arr(me[k], 'viol_hmax'), s1v_)
        col = COL['vq'] if k == 'vq_multi' else ('0.55' if k.startswith('s2') else '#1f77b4')
        ax.barh(i, m_, xerr=h_, color=col, height=0.6, error_kw=dict(lw=0.7, capsize=1.5))
        cr = 1.0 if k == 's2_full' else (0.125 if k == 's2_lite' else arr(me[k], 'compute_rate').mean())
        keys[i] = (k, f'{lab} [{cr:.2f}]')
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([l for _, l in keys]); ax.invert_yaxis()
    ax.axvline(0, color='k', lw=0.6); ax.set_xlabel('change in violations vs. S1 [%]')
    save(fig, 'fig_multi.pdf')
    fig, ax = plt.subplots(figsize=SZ)
    vv = np.linspace(0, 1, 200)
    for i, (lab, col) in enumerate([('light ($M$=4, $H$=16)', '#ff7f0e'), ('full ($M$=16, $H$=32)', COL['vq'])]):
        ax.plot(vv, np.interp(vv, mc['knots'][i], mc['vals'][i]), color=col, label=lab)
    ax.axhline(0, color='k', lw=0.6); ax.set_xlabel('risk signal $v$'); ax.set_ylabel('$\\hat g_j(v)$ (reference-measured)')
    ax.set_yscale('symlog', linthresh=0.5); ax.legend(frameon=False, loc='upper left')
    save(fig, 'fig_multi_maps.pdf')
    for k, tag in [('s2_lite', 'SLITE'), ('s2_full', 'SFULL'), ('vq_lite', 'VQLITE'), ('vq_full', 'VQFULL'), ('vq_multi', 'VQMULTI')]:
        m_, h_ = paired(arr(me[k], 'viol_hmax'), s1v_)
        N[f'MV{tag}'] = f"{-m_:.1f}"; N[f'MV{tag}CI'] = f"{h_:.1f}"
        N[f'MA{tag}'] = f"{paired(arr(me[k], 'aoi'), s1a_)[0]:+.1f}"
    for k, tag in [('vq_lite', 'VQLITE'), ('vq_full', 'VQFULL'), ('vq_multi', 'VQMULTI')]:
        N[f'MC{tag}'] = f"{arr(me[k], 'compute_rate').mean():.3f}"
    mix = np.mean([r['planner_mix'] for r in me['vq_multi']], axis=0)
    N['MIXNONE'] = f"{100*mix[0]:.1f}"; N['MIXLITE'] = f"{100*mix[1]:.1f}"; N['MIXFULL'] = f"{100*mix[2]:.1f}"
    g_ = np.array(mc['g']); N['NEGLITE'] = f"{100*np.mean(g_[:, 0] < 0):.1f}"; N['NEGFULL'] = f"{100*np.mean(g_[:, 1] < 0):.1f}"
    N['MEANLITE'] = f"{g_[:, 0].mean():.2f}"; N['MEANFULL'] = f"{g_[:, 1].mean():.2f}"
    N['MNSEEDS'] = str(len(me['s1']))

if os.path.exists(os.path.join(RES, 'dense_eval.json')):
    de = L('dense_eval')
    s1v_ = arr(de['s1'], 'viol_hmax'); s1a_ = arr(de['s1'], 'aoi')
    keys = [('random', 'random'), ('static', 'static threshold'), ('vq', 'virtual queue'), ('vq30', 'VQ, $\\rho$=0.3'), ('s2', 'always-S2')]
    keys = [kk for kk in keys if kk[0] in de]
    fig, ax = plt.subplots(figsize=SZ)
    for i, (k, lab) in enumerate(keys):
        m_, h_ = paired(arr(de[k], 'viol_hmax'), s1v_)
        ax.barh(i, m_, xerr=h_, color=COL.get(k, COL['vq']) if k != 's2' else '0.55', height=0.6, error_kw=dict(lw=0.7, capsize=1.5))
        rt = 1.0 if k == 's2' else arr(de[k], 's2_rate').mean()
        keys[i] = (k, f'{lab} [{rt:.2f}]')
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([l for _, l in keys]); ax.invert_yaxis()
    ax.axvline(0, color='k', lw=0.6); ax.set_xlabel('change in violations vs. S1 [%]')
    save(fig, 'fig_dense.pdf')
    for k, tag in [('random', 'RND'), ('static', 'STAT'), ('vq', 'VQ'), ('s2', 'STWO')]:
        m_, h_ = paired(arr(de[k], 'viol_hmax'), s1v_)
        N[f'DV{tag}'] = f"{-m_:.1f}"; N[f'DV{tag}CI'] = f"{h_:.1f}"
        N[f'DA{tag}'] = f"{paired(arr(de[k], 'aoi'), s1a_)[0]:+.1f}"
    N['DRVQ'] = f"{arr(de['vq'], 's2_rate').mean():.3f}"; N['DRSTAT'] = f"{arr(de['static'], 's2_rate').mean():.3f}"
    for nm, fn, pre in [('base', 'calibration', 'B'), ('dense', 'dense_calib', 'D')]:
        cc = L(fn); vv_ = np.array(cc['cal']['v'] if 'cal' in cc else cc['v']); gg_ = np.array(cc['cal']['g'] if 'cal' in cc else cc['g'])
        o_ = np.argsort(-vv_); cs_ = np.cumsum(gg_[o_]) / gg_.sum()
        N[f'{pre}HIGHV'] = f"{100*np.mean(vv_ >= 0.8):.0f}"; N[f'{pre}CAPTEN'] = f"{100*cs_[int(0.1*len(vv_))-1]:.0f}"
        N[f'{pre}BUDSEVENTY'] = f"{100*(np.argmax(cs_ >= 0.7)+1)/len(vv_):.0f}"
        N[f'{pre}TOPTEN'] = f"{100*np.sort(gg_)[::-1][:len(gg_)//10].sum()/gg_.sum():.0f}"
    if 'vq30' in de:
        m_, h_ = paired(arr(de['vq30'], 'viol_hmax'), s1v_); N['DVVQTHIRTY'] = f"{-m_:.1f}"; N['DVVQTHIRTYCI'] = f"{h_:.1f}"
        N['DRVQTHIRTY'] = f"{arr(de['vq30'], 's2_rate').mean():.3f}"
    N['DSONEVIOL'] = f"{100*s1v_.mean():.2f}"; N['DNSEEDS'] = str(len(de['s1']))

if os.path.exists(os.path.join(RES, 'design6.json')):
    d6 = L('design6'); s1c_ = arr(d6['S1'], 'cost'); s1v_ = arr(d6['S1'], 'viol_hmax')
    for pre, vals, fname, xl in [('M', [2, 4, 8, 16], 'fig_M.pdf', 'imagined rollouts $M$'),
                                 ('H', [8, 16, 32, 64], 'fig_H.pdf', 'planning horizon $H$')]:
        fig, ax = plt.subplots(figsize=SZ)
        ms, hs = [], []
        for x in vals:
            key = f'{pre}{x}' if not (pre == 'H' and x == 32) else 'M16'
            m_, h_ = paired(arr(d6[key], 'viol_hmax'), s1v_); ms.append(m_); hs.append(h_)
        ax.errorbar(vals, ms, yerr=hs, fmt='-o', color=COL['vq'], capsize=2, lw=1)
        ax.axhline(0, color='k', ls=':', lw=0.8, label='always-S1')
        ax.set_xscale('log', base=2); ax.set_xticks(vals); ax.set_xticklabels([str(x) for x in vals])
        ax.set_xlabel(xl); ax.set_ylabel('change in violations vs. S1 [%]'); ax.legend(frameon=False)
        save(fig, fname)
        for x, m_, h_ in zip(vals, ms, hs):
            N[f'DSG{pre}{["","A","B","C","D","E","F","G","H"][int(np.log2(x))]}'] = f"{-m_:.1f}"
            N[f'DSG{pre}{["","A","B","C","D","E","F","G","H"][int(np.log2(x))]}CI'] = f"{h_:.1f}"
    N['DSGCOSTMTWO'] = f"{paired(arr(d6['M2'], 'cost'), s1c_)[0]:+.1f}"
    N['DSGNSEEDS'] = str(len(d6['S1']))

if os.path.exists(os.path.join(RES, 'hmax.json')):
    hm = L('hmax'); rows = []
    mr_ = L('main_rho')
    for hf, src in [(1.5, hm), (2.0, None), (3.0, hm)]:
        if src is None:
            s1 = arr(mr_['s1'], 'viol_hmax')[:4]; st_ = arr(mr_['static_0.1'], 'viol_hmax')[:4]; vq_ = arr(mr_['vq_0.1'], 'viol_hmax')[:4]
        else:
            s1 = arr(hm[f's1_{hf}'], 'viol_hmax'); st_ = arr(hm[f'static_{hf}'], 'viol_hmax'); vq_ = arr(hm[f'vq_{hf}'], 'viol_hmax')
        a, ha = paired(st_, s1); b, hb = paired(vq_, s1)
        rows.append(f"{hf:g}$K$ & {100*s1.mean():.2f} & {a:+.1f} $\\pm$ {ha:.1f} & {b:+.1f} $\\pm$ {hb:.1f}\\\\")
    N['HMAXROWS'] = '\n'.join(rows)

if os.path.exists(os.path.join(RES, 'mismatch.json')):
    mm = L('mismatch'); d6 = L('design6'); s1v_ = arr(d6['S1'], 'viol_hmax')
    for k, tag in [('q0.01', 'LOW'), ('q100.0', 'HIGH')]:
        m_, h_ = paired(arr(mm[k], 'viol_hmax'), s1v_); N[f'MM{tag}'] = f"{-m_:.1f}"; N[f'MM{tag}CI'] = f"{h_:.1f}"
    m_, h_ = paired(arr(d6['M16'], 'viol_hmax'), s1v_); N['MMNOM'] = f"{-m_:.1f}"; N['MMNOMCI'] = f"{h_:.1f}"

with open(os.path.join(HERE, '..', 'paper', 'numbers.tex'), 'w') as f:
    for k, val in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(json.dumps({k: val for k, val in N.items() if 'ROWS' not in k}, indent=0))
