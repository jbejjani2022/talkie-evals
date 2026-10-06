"""Figures from analysis tables (run locally: uv run --no-project --with matplotlib --with numpy).

Identity encoding is fixed across every figure: Vintage = orange, Web = blue (as in the TRAIT deck);
SFT condition = marker shape (base: hollow circle, Tulu 3: filled circle, Vintage SFT: filled square).
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

COLOR = {'vintage': '#eb6834', 'web': '#2a78d6'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DIVERGING = ['#104281', '#3987e5', '#9ec5f4', '#f0efec', '#f4b0a6', '#e34948', '#8f1d1d']
MARKER = {'base': dict(marker='o', facecolor='white'), 'tulu': dict(marker='o'), 'vsft': dict(marker='s')}
LABEL = {'base': 'Base', 'tulu': 'Tulu 3 SFT', 'vsft': 'Vintage SFT'}
FAMILY = {'vintage': 'Vintage', 'web': 'Web'}
INTERFACE = {'bare': 'bare interface (Question/Answer)', 'chat': 'chat interface (<|user|>/<|assistant|>)'}
ARM_ORDER = ['vintage-base', 'vintage-tulu', 'vintage-vsft', 'web-base', 'web-tulu', 'web-vsft']
TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism',
          'machiavellianism', 'narcissism', 'psychopathy']


def read(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ('True', 'False'):
                r[k] = v == 'True'
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'legend.frameon': False, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'axes.titlecolor': INK, 'axes.axisbelow': True})
    return plt


def save(fig, output, name, note=None):
    import matplotlib.pyplot as plt
    if note:
        fig.text(0.01, -0.01, note, fontsize=7, color=MUTED, ha='left', va='top', wrap=True)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def arm_parts(arm):
    family, sft = arm.split('-')
    return family, sft


def point(ax, x, y, arm, size=34, xerr=None, yerr=None, zorder=3, alpha=1.0):
    family, sft = arm_parts(arm)
    style = MARKER[sft]
    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor=COLOR[family], elinewidth=1, alpha=0.6 * alpha, zorder=zorder - 1)
    ax.scatter([x], [y], s=size, marker=style['marker'], facecolor=style.get('facecolor', COLOR[family]),
               edgecolor=COLOR[family], linewidth=1.4, zorder=zorder, alpha=alpha)


def legend_handles(arms=ARM_ORDER):
    from matplotlib.lines import Line2D
    handles = []
    for arm in arms:
        family, sft = arm_parts(arm)
        style = MARKER[sft]
        handles.append(Line2D([], [], linestyle='none', marker=style['marker'], markersize=6,
            markerfacecolor=style.get('facecolor', COLOR[family]), markeredgecolor=COLOR[family], markeredgewidth=1.4,
            label=f'{FAMILY[family]} · {LABEL[sft]}'))
    return handles


def top_legend(fig, arms=ARM_ORDER, y=1.02):
    fig.legend(handles=legend_handles(arms), loc='lower center', ncol=len(arms), bbox_to_anchor=(0.5, y), fontsize=7)


def ci(row, key='value'):
    return [[row[key] - row['ci_low']], [row['ci_high'] - row[key]]]


SHORT = {'base': 'base', 'tulu': 'Tulu', 'vsft': 'V-SFT'}


def arm_ticks(ax, arms, axis='x'):
    labels = [f'{FAMILY[arm_parts(a)[0]]}\n{SHORT[arm_parts(a)[1]]}' for a in arms]
    (ax.set_xticks if axis == 'x' else ax.set_yticks)(range(len(arms)), labels, fontsize=7)


def select(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


# ---------------------------------------------------------------- 1. format controls
def fig_format(t, output):
    plt = setup()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={'width_ratios': [1, 1, 1.25, 1.25]})
    for ax, scoring in zip(axes[:2], ('letter', 'text')):
        for k, interface in enumerate(('bare', 'chat')):
            for i, arm in enumerate(ARM_ORDER):
                r = select(t['arc_easy'], interface=interface, scoring=scoring, arm=arm)[0]
                point(ax, i + (k - 0.5) * 0.28, r['accuracy'], arm, yerr=ci(r, 'accuracy'), alpha=1 if interface == 'chat' else 0.45)
        ax.axhline(25, color=MUTED, linewidth=1); ax.text(5.4, 26, 'chance', color=MUTED, fontsize=7, ha='right')
        arm_ticks(ax, ARM_ORDER); ax.set_ylim(15, 65); ax.set_ylabel('ARC-Easy accuracy (%)')
        ax.set_title({'letter': 'Answer by letter "(A)…"', 'text': 'Answer by option text (TRAIT-style)'}[scoring])
    axes[0].text(0.02, 0.97, 'faded = bare · solid = chat', transform=axes[0].transAxes, fontsize=7, color=MUTED, va='top')
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#9ec5f4', '#3987e5', '#104281'])
    ax = axes[2]
    cols = [('ai_risk', 'letter'), ('ai_risk', 'text'), ('global_opinions', 'letter'), ('global_opinions', 'text')]
    keys = [(i, a) for i in ('bare', 'chat') for a in ARM_ORDER]
    grid = np.array([[select(t['order_invariance'], interface=i, arm=a, eval=e, scoring=s)[0]['order_invariance_pct']
                      for e, s in cols] for i, a in keys])
    heat(ax, grid, cmap, [f'{e.split("_")[0].replace("global", "GOQA").replace("ai", "AI risk")}\n{s}' for e, s in cols],
         [f'{i} · {a}' for i, a in keys], 'Answers follow content when options are reordered (%)')
    ax = axes[3]
    evals = ['persona', 'sycophancy', 'ai_risk', 'winogenerated', 'global_opinions', 'arc_easy']
    grid = np.array([[select(t['diagnostics'], interface=i, arm=a, eval=e)[0]['candidate_mass_pct'] for e in evals] for i, a in keys])
    heat(ax, grid, cmap, ['persona', 'syco-\nphancy', 'AI risk', 'wino', 'GOQA', 'ARC'], [''] * len(keys),
         'Next-token probability on the answer options (%)')
    fig.tight_layout()
    save(fig, output, 'fig1_format_controls',
         'Left: Vintage arms are at chance when answering by letter (they choose "(A)" almost always), and only modestly above chance with '
         'option-text scoring; ARC also requires modern science knowledge. Middle: AI risk = same option content chosen in both orders; GOQA = '
         'similarity of original- and reversed-order answer distributions. Right: low mass means the preference is read from a small part of the '
         'distribution. Letter-mode results are therefore not interpreted for Vintage arms.')


def heat(ax, grid, cmap, xlabels, ylabels, title):
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    for (y, x), v in np.ndenumerate(grid):
        ax.text(x, y, f'{v:.0f}', ha='center', va='center', fontsize=6.5, color='white' if v > 60 else INK)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=6.5)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=6.5)
    ax.tick_params(length=0)
    ax.grid(False); ax.set_title(title, fontsize=9)
    ax.axhline(5.5, color=SURFACE, linewidth=3)


# ---------------------------------------------------------------- 2-3. persona
def persona_table(t, interface, metric):
    v = defaultdict(dict)
    if metric == 'soft':
        for r in select(t['persona'], interface=interface):
            v[r['behavior']][r['arm']] = r['value']
    else:
        for r in select(t['persona_extra'], interface=interface):
            v[r['behavior']][r['arm']] = r['calibrated_balanced']
    return v


def fig_persona_shift(t, output, interface, label_n=4):
    plt = setup()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for row, metric in enumerate(('soft', 'argmax')):
        v = persona_table(t, interface, metric)
        behaviors = sorted(v)
        get = lambda arm: np.array([v[b][arm] for b in behaviors])
        vb, vt, wb, wt = get('vintage-base'), get('vintage-tulu'), get('web-base'), get('web-tulu')
        panels = [(vb, wb, 'Vintage base (%)', 'Web base (%)', 'Before SFT: do the bases answer alike?', True),
                  (wt - wb, vt - vb, 'Web: Tulu − base (pp)', 'Vintage: Tulu − base (pp)', 'Does Tulu move both bases the same way?', False),
                  (wb - vb, vt - vb, 'Pre-SFT gap: Web − Vintage (pp)', 'Vintage: Tulu − base (pp)', 'Does Tulu move Vintage toward Web?', False)]
        for ax, (x, y, xl, yl, title, levels) in zip(axes[row], panels):
            ax.scatter(x, y, s=12, color=MUTED, alpha=0.5, linewidth=0)
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, zorder=0)
            if not levels:
                ax.axhline(0, color=GRID, linewidth=1); ax.axvline(0, color=GRID, linewidth=1)
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.03, 0.97, f'r = {r:.2f} · slope = {np.polyfit(x, y, 1)[0]:.2f}\nmean |x| = {np.abs(x).mean():.1f} · mean |y| = {np.abs(y).mean():.1f}'
                    if not levels else f'r = {r:.2f}', transform=ax.transAxes, va='top', fontsize=8, color=INK)
            far = np.argsort(-np.abs(y - x) if levels else -np.abs(y))[:label_n]
            for i in far:
                ax.annotate(behaviors[i], (x[i], y[i]), fontsize=5.5, color=INK, xytext=(3, 3), textcoords='offset points')
                ax.scatter([x[i]], [y[i]], s=14, color=INK, linewidth=0)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(title if row == 0 else '')
        axes[row, 1].text(0.5, 1.13 if row == 0 else 1.04, {'soft': 'Metric: expected P(matching answer)',
                          'argmax': 'Metric: calibrated argmax choice'}[metric], transform=axes[row, 1].transAxes,
                          ha='center', fontsize=10, fontweight='bold', color=MUTED)
    fig.tight_layout()
    save(fig, output, f'fig2_persona_shift_{interface}',
         f'Persona, 135 behaviors · {INTERFACE[interface]} · one point per behavior. Top: expected probability of the behavior-matching answer, '
         'balanced over Yes/No polarity (50 = no preference); it shrinks toward 50 for less confident models. Bottom: share of statements where '
         'the model picks the matching answer after removing its average Yes/No bias. Claims in findings.md hold under both rows.')


def fig_persona_dots(t, output, interface, categories, name, metric='argmax'):
    plt = setup()
    from .metrics import persona_category
    v = persona_table(t, interface, metric)
    behaviors = [b for c in categories for b in sorted(v, key=lambda b: -v[b]['web-tulu']) if persona_category(b) == c]
    height = 0.2 * len(behaviors) + 1.6
    fig, axes = plt.subplots(1, 2, figsize=(11, height), sharey=True)
    for ax, family in zip(axes, ('vintage', 'web')):
        for i, b in enumerate(behaviors):
            base, tulu = v[b][f'{family}-base'], v[b][f'{family}-tulu']
            ax.plot([base, tulu], [i, i], color=COLOR[family], linewidth=1.2, alpha=0.45, zorder=1)
            point(ax, base, i, f'{family}-base', size=20)
            point(ax, tulu, i, f'{family}-tulu', size=20)
            point(ax, v[b][f'{family}-vsft'], i, f'{family}-vsft', size=12, alpha=0.7)
        ax.axvline(50, color=MUTED, linewidth=0.8)
        ax.set_xlim(-2, 102); ax.set_title(FAMILY[family])
        ax.set_xlabel('% of statements answered in the behavior-matching way')
        start = 0
        for c in categories:
            n = sum(persona_category(b) == c for b in behaviors)
            if start:
                ax.axhline(start - 0.5, color=MUTED, linewidth=0.6)
            start += n
    axes[0].set_yticks(range(len(behaviors)), behaviors, fontsize=6.5); axes[0].invert_yaxis()
    start = 0
    for c in categories:
        n = sum(persona_category(b) == c for b in behaviors)
        axes[1].text(104, start + n / 2 - 0.5, c, fontsize=8, color=INK, va='center', fontweight='bold')
        start += n
    top_legend(fig, y=1.0)
    fig.tight_layout()
    save(fig, output, f'{name}_{interface}', f'Persona · {INTERFACE[interface]} · calibrated argmax metric · lines connect base → Tulu 3 SFT; '
         'small squares = Vintage SFT control. Rows sorted by Web-Tulu within each Perez et al. category.')


def fig_persona_heatmap(t, output, interface):
    plt = setup()
    from .metrics import PERSONA_CATEGORIES, persona_category
    d = defaultdict(dict)
    for r in select(t['persona_deltas'], interface=interface):
        d[r['behavior']][f"{r['family']}-{r['sft']}"] = r
    categories = list(PERSONA_CATEGORIES) + ['Advanced AI risk']
    behaviors = [b for c in categories for b in sorted(d) if persona_category(b) == c]
    cols = ['vintage-tulu', 'web-tulu', 'vintage-vsft', 'web-vsft']
    grid = np.array([[d[b][c]['delta'] for c in cols] for b in behaviors])
    sig = np.array([[d[b][c]['ci_low'] > 0 or d[b][c]['ci_high'] < 0 for c in cols] for b in behaviors])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', DIVERGING[::-1])
    lim = np.ceil(np.abs(grid).max() / 5) * 5
    fig, ax = plt.subplots(figsize=(6, 0.13 * len(behaviors) + 1.5))
    im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
    for (y, x), val in np.ndenumerate(grid):
        if sig[y, x]:
            ax.text(x, y, f'{val:+.0f}', ha='center', va='center', fontsize=4.8, color=INK)
    ax.set_xticks(range(len(cols)), ['Vintage\nTulu 3', 'Web\nTulu 3', 'Vintage\nVintage SFT', 'Web\nVintage SFT'], fontsize=7)
    ax.set_yticks(range(len(behaviors)), behaviors, fontsize=4.8); ax.grid(False)
    start = 0
    for c in categories:
        n = sum(persona_category(b) == c for b in behaviors)
        ax.axhline(start - 0.5, color=SURFACE, linewidth=2)
        ax.text(len(cols) - 0.35, start + n / 2 - 0.5, c, fontsize=6, color=INK, va='center')
        start += n
    fig.colorbar(im, ax=ax, shrink=0.25, label='SFT − base (pp)', pad=0.4)
    ax.set_title(f'Persona: SFT − base, all behaviors · {interface}')
    save(fig, output, f'figA_persona_heatmap_{interface}', 'Expected-probability metric · numbers only where the paired 95% bootstrap interval excludes 0.')


# ---------------------------------------------------------------- 4. sycophancy
def fig_sycophancy(t, output, scoring='text'):
    plt = setup()
    subsets = [('philpapers2020', 'PhilPapers'), ('nlp_survey', 'NLP survey'), ('political_typology_quiz', 'Political typology')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey='row')
    for row, interface in enumerate(('bare', 'chat')):
        for col, (s, title) in enumerate(subsets):
            ax = axes[row, col]
            for i, arm in enumerate(ARM_ORDER):
                r = select(t['sycophancy'], interface=interface, scoring=scoring, subset=s, measure='excess_soft', arm=arm)[0]
                point(ax, i, r['value'], arm, yerr=ci(r))
            ax.axhline(0, color=MUTED, linewidth=0.8)
            arm_ticks(ax, ARM_ORDER); ax.set_title(f'{title} · {interface}')
            if col == 0:
                ax.set_ylabel('Agreement with the user beyond\nthe bio-blind baseline (pp)')
    fig.tight_layout()
    save(fig, output, f'fig4_sycophancy_{scoring}', f'Sycophancy · {scoring} scoring · expected agreement with the view stated in the user\'s bio, minus the '
         'agreement expected if the answer ignored the bio (the same question under all bios). 95% cluster bootstrap over questions '
         '(109 PhilPapers, 32 NLP, 15 political). ' + ('Letter scoring is not interpretable for Vintage arms (see Fig. 1).' if scoring == 'letter' else ''))


# ---------------------------------------------------------------- 5. AI risk
def fig_ai_risk(t, output, interface, scoring='text', source='human'):
    plt = setup()
    rows = select(t['ai_risk'], interface=interface, scoring=scoring, source=source)
    control = {(r['subset'], r['arm']): r['value'] for r in select(t.get('ai_risk_options_only', []), interface=interface, source=source)}
    subsets = sorted({r['subset'] for r in rows}, key=lambda s: select(rows, subset=s, arm='web-tulu')[0]['value'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(subsets) + 1.6), sharey=True)
    for ax, family in zip(axes, ('vintage', 'web')):
        for i, s in enumerate(subsets):
            get = {r['arm']: r for r in rows if r['subset'] == s}
            base, tulu = get[f'{family}-base'], get[f'{family}-tulu']
            ax.plot([base['value'], tulu['value']], [i, i], color=COLOR[family], linewidth=1.2, alpha=0.45)
            point(ax, base['value'], i, f'{family}-base', size=22, xerr=ci(base))
            point(ax, tulu['value'], i, f'{family}-tulu', size=22, xerr=ci(tulu))
            point(ax, get[f'{family}-vsft']['value'], i, f'{family}-vsft', size=12, alpha=0.7)
            if control and scoring == 'text':
                ax.scatter([control[s, f'{family}-tulu']], [i], marker='|', s=60, color=COLOR[family], alpha=0.8, zorder=2)
        ax.axvline(50, color=MUTED, linewidth=0.8); ax.set_xlim(15, 90)
        ax.set_title(FAMILY[family]); ax.set_xlabel('% answers showing the named behavior (50 = indifferent)')
    axes[0].set_yticks(range(len(subsets)), [s.split('/')[1] for s in subsets], fontsize=8)
    top_legend(fig, y=1.0)
    fig.tight_layout()
    save(fig, output, f'fig5_ai_risk_{source}_{scoring}_{interface}', f'Advanced AI risk ({source}-written) · {scoring} scoring · {INTERFACE[interface]} · '
         'averaged over every option order · 95% bootstrap. | = Tulu arm scored on the options alone, without the question (option-wording prior). '
         'Survival, power and wealth (human-written) are re-oriented: their source labels mark the safe answer.')


# ---------------------------------------------------------------- 6. winogenerated
def fig_winogenerated(t, output, interface):
    plt = setup()
    occ = select(t['winogenerated_occupations'], interface=interface)
    summ = {r['arm']: r for r in select(t['winogenerated'], interface=interface)}
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2), gridspec_kw={'width_ratios': [1, 1, 1, 0.85]})
    for arm in ARM_ORDER:
        family, sft = arm_parts(arm)
        ax = axes[0 if family == 'vintage' else 1, ['base', 'tulu', 'vsft'].index(sft)]
        rows = [r for r in occ if r['arm'] == arm]
        x = np.array([r['bls_percent_women'] for r in rows]); y = np.array([100 * r['p_female'] / (r['p_female'] + r['p_male']) for r in rows])
        ax.scatter(x, y, s=9, color=COLOR[family], alpha=0.55, linewidth=0)
        ax.plot([0, 100], [0, 100], color=GRID, linewidth=1, zorder=0)
        s = summ[arm]
        ax.text(0.03, 0.97, f"r = {s['r']:.2f} [{s['r_ci_low']:.2f}, {s['r_ci_high']:.2f}]\nmean she-share {s['female_share_pct']:.0f}%",
                transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_title(f'{FAMILY[family]} · {LABEL[sft]}')
        ax.set_xlabel('BLS 2019 % women in occupation'); ax.set_ylabel('P(she) / (P(he) + P(she)) (%)')
    for row, family in enumerate(('vintage', 'web')):
        ax = axes[row, 3]
        sel = [a for a in ARM_ORDER if a.startswith(family)]
        for i, a in enumerate(sel):
            s = summ[a]
            point(ax, i, s['neutral_pct'], a, yerr=[[s['neutral_pct'] - s['neutral_ci_low']], [s['neutral_ci_high'] - s['neutral_pct']]])
        ax.set_xticks(range(len(sel)), [LABEL[arm_parts(a)[1]] for a in sel], fontsize=7)
        ax.set_ylim(0, 45); ax.set_title(f'{FAMILY[family]} · singular "they"'); ax.set_ylabel('Mean P(they / their / them) (%)')
    fig.tight_layout()
    save(fig, output, f'fig6_winogenerated_{interface}', f'Winogenerated (299 occupations × 10 sentences) · {INTERFACE[interface]} · one point per occupation · '
         'r = Pearson correlation with BLS 2019 · 95% bootstrap over occupations. BLS statistics are modern; a 1930 census comparison is a suggested follow-up.')


# ---------------------------------------------------------------- 7. global opinions
def fig_global_opinions(t, output, interface, scoring='text', calibrated=False, metric='alignment', min_questions=200, show=10):
    plt = setup()
    rows = [r for r in select(t['global_opinions'], interface=interface, scoring=scoring, calibrated=calibrated, metric=metric)
            if r['country'] != '_temperature' and r['questions'] >= min_questions]
    deltas = [r for r in select(t['global_opinions_deltas'], interface=interface, scoring=scoring, calibrated=calibrated, metric=metric)
              if r['questions'] >= min_questions]
    unit = 'alignment, r × 100' if metric == 'alignment' else '1 − JS distance, %'
    sim = defaultdict(dict)
    for r in rows:
        sim[r['country']][r['arm']] = r
    vmw = {r['country']: r for r in deltas if r['family'] == 'vintage-minus-web' and r['sft'] == 'base'}
    vmw_t = {r['country']: r for r in deltas if r['family'] == 'vintage-minus-web' and r['sft'] == 'tulu'}
    order = sorted(vmw, key=lambda c: vmw[c]['delta'])
    picked = order[:show] + order[-show:]
    height = 0.26 * len(picked) + 1.8
    fig, axes = plt.subplots(1, 3, figsize=(14.5, height), gridspec_kw={'width_ratios': [1.15, 1, 1]})
    ax = axes[0]
    for i, c in enumerate(picked):
        ax.plot([sim[c]['vintage-base']['value'], sim[c]['web-base']['value']], [i, i], color=MUTED, linewidth=0.8, zorder=1)
        for arm in ('vintage-base', 'web-base'):
            point(ax, sim[c][arm]['value'], i, arm, size=20)
    ax.set_yticks(range(len(picked)), [f"{c} ({int(sim[c]['vintage-base']['questions'])})" for c in picked], fontsize=7)
    ax.axhline(show - 0.5, color=MUTED, linewidth=0.6); ax.invert_yaxis()
    ax.set_xlabel(f'Match to the country\'s answers ({unit})')
    ax.set_title('Base models: the most Web-leaning (top)\nand most Vintage-leaning (bottom) countries')
    ax = axes[1]
    for i, c in enumerate(picked):
        point(ax, vmw[c]['delta'], i - 0.15, 'vintage-base', size=18, xerr=ci(vmw[c], 'delta'))
        if c in vmw_t:
            point(ax, vmw_t[c]['delta'], i + 0.15, 'vintage-tulu', size=18, xerr=ci(vmw_t[c], 'delta'))
    ax.axvline(0, color=MUTED, linewidth=0.8); ax.set_yticks(range(len(picked)), [''] * len(picked))
    ax.axhline(show - 0.5, color=MUTED, linewidth=0.6); ax.invert_yaxis()
    ax.set_xlabel(f'Vintage − Web ({unit})'); ax.set_title('Vintage − Web before (hollow)\nand after Tulu (filled)')
    ax = axes[2]
    tulu = {(r['family'], r['country']): r for r in deltas if r['sft'] == 'tulu' and r['family'] in ('vintage', 'web')}
    countries = sorted({c for _, c in tulu})
    x = np.array([tulu['web', c]['delta'] for c in countries]); y = np.array([tulu['vintage', c]['delta'] for c in countries])
    ax.scatter(x, y, s=12, color=MUTED, alpha=0.6, linewidth=0)
    for i in np.argsort(-np.abs(y))[:8]:
        ax.annotate(countries[i], (x[i], y[i]), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
    ax.text(0.03, 0.97, f'r = {np.corrcoef(x, y)[0, 1]:.2f}', transform=ax.transAxes, va='top', fontsize=8)
    ax.set_xlabel(f'Web: Tulu − base ({unit})'); ax.set_ylabel(f'Vintage: Tulu − base ({unit})')
    ax.set_title(f'Tulu effect per country\n({len(countries)} countries with ≥{min_questions} questions)')
    top_legend(fig, ['vintage-base', 'vintage-tulu', 'web-base'], y=1.0)
    fig.tight_layout()
    temps = {r['arm']: r['value'] for r in select(t['global_opinions'], interface=interface, scoring=scoring, calibrated=True, country='_temperature')}
    name = 'alignment' if metric == 'alignment' else ('calibrated' if calibrated else 'raw')
    save(fig, output, f'fig7_global_opinions_{scoring}_{name}_{interface}',
         f'GlobalOpinionQA · {scoring} scoring · {INTERFACE[interface]} · model distribution averaged over original/reversed option order' +
         (' · alignment = correlation between the model\'s log-probabilities and the country\'s answer shares across options (questions with '
          '≥3 options); unaffected by how confident the model is' if metric == 'alignment' else '') +
         (', then sharpness-matched with one temperature per model fitted to the country-pooled answers (T: ' +
          ', '.join(f'{a} {v:.2f}' for a, v in temps.items()) + ')' if calibrated else '') +
         ' · 95% paired bootstrap over questions. Without calibration a uniform guesser is more similar to every country than any model.')


# ---------------------------------------------------------------- 8. convergent validity with TRAIT
def fig_convergent(t, output, interface):
    plt = setup()
    trait = {(r['arm'], r['trait']): r['trait_rate'] for r in select(t['trait_reference'], interface=interface)}
    soft = {(r['arm'], r['behavior']): r['value'] for r in select(t['persona'], interface=interface) if r['behavior'] in TRAITS}
    hard = {(r['arm'], r['behavior']): r['calibrated_balanced'] for r in select(t['persona_extra'], interface=interface) if r['behavior'] in TRAITS}
    short = dict(zip(TRAITS, ['O', 'C', 'E', 'A', 'N', 'Mach', 'Narc', 'Psych']))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    xs, ys = [], []
    for arm in ARM_ORDER:
        for tr in TRAITS:
            x, y = trait[arm, tr], hard[arm, tr]
            xs.append(x); ys.append(y)
            point(ax, x, y, arm, size=22)
    ax.text(0.03, 0.97, f'r = {np.corrcoef(xs, ys)[0, 1]:.2f} (48 points)', transform=ax.transAxes, va='top', fontsize=8)
    ax.set_xlabel('TRAIT: high-trait answer chosen (%)'); ax.set_ylabel('Persona: trait statements endorsed (%)')
    ax.set_title('Same 8 traits, two instruments')
    ax = axes[1]
    for k, (title, a_arm, b_arm) in enumerate([('Web − Vintage (base)', 'web-base', 'vintage-base')]):
        x = [trait[a_arm, tr] - trait[b_arm, tr] for tr in TRAITS]
        for y_src, mk, lbl in ((soft, 'o', 'persona: expected P'), (hard, 's', 'persona: argmax')):
            y = [y_src[a_arm, tr] - y_src[b_arm, tr] for tr in TRAITS]
            ax.scatter(x, y, marker=mk, s=28, facecolor='white' if mk == 'o' else MUTED, edgecolor=MUTED, label=lbl)
            for tr, xi, yi in zip(TRAITS, x, y):
                ax.annotate(short[tr], (xi, yi), fontsize=6, xytext=(3, 2), textcoords='offset points')
    lim = 30
    ax.plot([-lim, lim], [-lim, lim], color=GRID); ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
    ax.set_xlabel('TRAIT gap (pp)'); ax.set_ylabel('Persona gap (pp)'); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Does persona reproduce TRAIT\'s Web−Vintage gap?')
    ax = axes[2]
    for arm in ('vintage-tulu', 'web-tulu', 'vintage-vsft', 'web-vsft'):
        base = arm.split('-')[0] + '-base'
        x = [trait[arm, tr] - trait[base, tr] for tr in TRAITS]; y = [hard[arm, tr] - hard[base, tr] for tr in TRAITS]
        for tr, xi, yi in zip(TRAITS, x, y):
            point(ax, xi, yi, arm, size=22)
        agree = sum(np.sign(a) == np.sign(b) for a, b in zip(x, y))
        ax.text(0.03, 0.97 - 0.06 * ['vintage-tulu', 'web-tulu', 'vintage-vsft', 'web-vsft'].index(arm),
                f'{FAMILY[arm.split("-")[0]]} {LABEL[arm.split("-")[1]]}: same sign {agree}/8', transform=ax.transAxes, va='top', fontsize=7,
                color=COLOR[arm.split('-')[0]])
    ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
    ax.set_xlabel('TRAIT: SFT − base (pp)'); ax.set_ylabel('Persona (argmax): SFT − base (pp)')
    ax.set_title('Do SFT shifts agree across instruments?')
    top_legend(fig, y=1.0)
    fig.tight_layout()
    save(fig, output, f'fig8_convergent_trait_{interface}', f'{INTERFACE[interface]} · TRAIT = bundled reference runs (B200; 2,000 answer pairs per trait) · '
         'persona = the 8 matching Perez et al. behaviors (1,000 statements each). TRAIT asks which of two actions the model would take; '
         'persona asks whether it would say a self-description.')


def load_tables(root):
    return {p.stem: read(p) for p in Path(root).glob('*.csv')}


def main(root, output, plausible_root=None, counts=None):
    t = load_tables(root)
    output = Path(output)
    if plausible_root:
        p = load_tables(plausible_root)
        for interface in ('bare', 'chat'):
            fig_anachronism(t, p, read(counts), output, interface)
    fig_format(t, output)
    for interface in ('bare', 'chat'):
        fig_persona_shift(t, output, interface)
        fig_persona_dots(t, output, interface, ['Personality: primary traits', 'Politics', 'Religion', 'Ethics'], 'fig3_persona_values')
        fig_persona_dots(t, output, interface, ['Personality: other traits', 'Beliefs', 'Advanced AI risk'], 'figB_persona_other')
        fig_persona_heatmap(t, output, interface)
        for scoring in ('text', 'letter'):
            fig_ai_risk(t, output, interface, scoring)
        fig_ai_risk(t, output, interface, 'text', 'lm')
        fig_winogenerated(t, output, interface)
        fig_global_opinions(t, output, interface, 'text', metric='alignment')
        fig_global_opinions(t, output, interface, 'letter', metric='alignment')
        fig_global_opinions(t, output, interface, 'text', False, 'similarity')
        fig_global_opinions(t, output, interface, 'text', True, 'similarity')
        fig_convergent(t, output, interface)
    for scoring in ('text', 'letter'):
        fig_sycophancy(t, output, scoring)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])


# ---------------------------------------------------------------- 9. anachronism second pass
def fig_anachronism(full, plaus, counts, output, interface='chat'):
    plt = setup()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))

    def diag(ax, xs, ys, title, xl='All items', yl='Historically plausible items only'):
        lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, zorder=0)
        ax.set_title(title, fontsize=9.5); ax.set_xlabel(xl); ax.set_ylabel(yl)

    ax = axes[0, 0]
    rows = [r for r in counts if r['subset'] == 'ALL']
    for i, r in enumerate(rows):
        ax.barh(i, r['anachronistic_pct'], color=MUTED, height=0.6)
        ax.text(r['anachronistic_pct'] + 1, i, f"{r['anachronistic_pct']:.0f}%", va='center', fontsize=8)
    ax.set_yticks(range(len(rows)), [r['eval'] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel('Items labelled anachronistic (%)'); ax.set_title('What the judge removes')

    ax = axes[0, 1]  # persona: behavior-level values per arm (argmax)
    f = {(r['arm'], r['behavior']): r['calibrated_balanced'] for r in select(full['persona_extra'], interface=interface)}
    p = {(r['arm'], r['behavior']): r['calibrated_balanced'] for r in select(plaus['persona_extra'], interface=interface)}
    xs, ys = [], []
    for k in sorted(set(f) & set(p)):
        if k[0] in ('vintage-base', 'vintage-tulu', 'web-base', 'web-tulu'):
            point(ax, f[k], p[k], k[0], size=10, alpha=0.6); xs.append(f[k]); ys.append(p[k])
    diag(ax, xs, ys, f'Persona behaviors (argmax) · r = {np.corrcoef(xs, ys)[0, 1]:.2f}')

    ax = axes[0, 2]  # persona: SFT shifts
    fd = {(r['family'], r['sft'], r['behavior']): r['delta'] for r in select(full['persona_deltas'], interface=interface)}
    pd_ = {(r['family'], r['sft'], r['behavior']): r['delta'] for r in select(plaus['persona_deltas'], interface=interface)}
    xs, ys = [], []
    for k in sorted(set(fd) & set(pd_)):
        if k[1] == 'tulu':
            point(ax, fd[k], pd_[k], f'{k[0]}-tulu', size=10, alpha=0.6); xs.append(fd[k]); ys.append(pd_[k])
    diag(ax, xs, ys, f'Persona Tulu shifts (expected P) · r = {np.corrcoef(xs, ys)[0, 1]:.2f}', 'All items (pp)', 'Plausible only (pp)')

    ax = axes[1, 0]  # sycophancy
    xs, ys = [], []
    for r in select(full['sycophancy'], interface=interface, scoring='text', measure='excess_soft'):
        q = select(plaus['sycophancy'], interface=interface, scoring='text', measure='excess_soft', subset=r['subset'], arm=r['arm'])
        if q:
            point(ax, r['value'], q[0]['value'], r['arm'], size=26); xs.append(r['value']); ys.append(q[0]['value'])
            if r['arm'] == 'web-tulu':
                ax.annotate({'philpapers2020': 'PhilPapers', 'political_typology_quiz': 'political', 'nlp_survey': 'NLP'}[r['subset']],
                            (r['value'], q[0]['value']), fontsize=7, xytext=(4, -8), textcoords='offset points')
    diag(ax, xs, ys, 'Sycophancy excess (text) · NLP survey fully removed', 'All items (pp)', 'Plausible only (pp)')

    ax = axes[1, 1]  # AI risk
    xs, ys = [], []
    for r in select(full['ai_risk'], interface=interface, scoring='text'):
        q = select(plaus['ai_risk'], interface=interface, scoring='text', subset=r['subset'], arm=r['arm'])
        if q and r['arm'] in ('vintage-base', 'vintage-tulu', 'web-base', 'web-tulu'):
            point(ax, r['value'], q[0]['value'], r['arm'], size=16, alpha=0.8); xs.append(r['value']); ys.append(q[0]['value'])
    diag(ax, xs, ys, f'AI-risk behaviors (text) · r = {np.corrcoef(xs, ys)[0, 1]:.2f}')

    ax = axes[1, 2]  # global opinions: vintage - web per country
    key = lambda rows: {(r['country'], r['sft']): r['delta'] for r in rows if r['family'] == 'vintage-minus-web' and r['questions'] >= 100}
    fg = key(select(full['global_opinions_deltas'], interface=interface, scoring='text', metric='alignment'))
    pg = key(select(plaus['global_opinions_deltas'], interface=interface, scoring='text', metric='alignment'))
    xs, ys = [], []
    for k in sorted(set(fg) & set(pg)):
        if k[1] in ('base', 'tulu'):
            point(ax, fg[k], pg[k], f'vintage-{k[1]}', size=12, alpha=0.7); xs.append(fg[k]); ys.append(pg[k])
    diag(ax, xs, ys, f'GOQA Vintage − Web per country · r = {np.corrcoef(xs, ys)[0, 1]:.2f}', 'All questions (r × 100)', 'Plausible only (r × 100)')
    top_legend(fig, ['vintage-base', 'vintage-tulu', 'web-base', 'web-tulu'], y=1.0)
    fig.tight_layout()
    save(fig, output, f'fig9_anachronism_{interface}', f'{INTERFACE[interface]} · second pass restricted to items the judge labelled historically '
         'plausible before 1931 (explicit modern references removed; UNCERTAIN excluded). Points on the diagonal are unchanged by filtering. '
         'GOQA: text scoring, alignment metric, countries with ≥100 questions in both passes; hollow = base, filled = after Tulu.')
