"""Figures from analysis tables (run locally: uv run --with matplotlib --with numpy).

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


def read(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
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
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=MUTED, ha='left', va='bottom')
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=200, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def arm_parts(arm):
    family, sft = arm.split('-')
    return family, sft


def point(ax, x, y, arm, size=34, xerr=None, yerr=None, label=None, zorder=3):
    family, sft = arm_parts(arm)
    style = MARKER[sft]
    face = style.get('facecolor', COLOR[family])
    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor=COLOR[family], elinewidth=1, alpha=0.7, zorder=zorder - 1)
    ax.scatter([x], [y], s=size, marker=style['marker'], facecolor=face, edgecolor=COLOR[family], linewidth=1.4,
               zorder=zorder, label=label)


def legend_handles():
    from matplotlib.lines import Line2D
    handles = []
    for family in ('vintage', 'web'):
        for sft in ('base', 'tulu', 'vsft'):
            style = MARKER[sft]
            handles.append(Line2D([], [], linestyle='none', marker=style['marker'], markersize=6,
                markerfacecolor=style.get('facecolor', COLOR[family]), markeredgecolor=COLOR[family], markeredgewidth=1.4,
                label=f'{FAMILY[family]} · {LABEL[sft]}'))
    return handles


def ci(row, key='value'):
    return [[row[key] - row['ci_low']], [row['ci_high'] - row[key]]]


# ---------------------------------------------------------------- format controls
def fig_format(tables, output):
    plt = setup()
    arc = tables['arc_easy']; diag = tables['diagnostics']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={'width_ratios': [1, 1, 1.6]})
    for ax, interface in zip(axes[:2], ('bare', 'chat')):
        rows = [r for r in arc if r['interface'] == interface]
        for i, r in enumerate(rows):
            point(ax, i, r['accuracy'], r['arm'], yerr=ci(r, 'accuracy'))
        ax.axhline(rows[0]['chance'], color=MUTED, linewidth=1)
        ax.text(len(rows) - 0.5, rows[0]['chance'] + 1, 'chance', color=MUTED, fontsize=7, ha='right')
        ax.set_xticks(range(len(rows)), [r['arm'].replace('-', '\n') for r in rows], fontsize=7)
        ax.set_ylim(0, 100); ax.set_ylabel('ARC-Easy accuracy (%)'); ax.set_title(f'Letter-format control · {interface}')
    ax = axes[2]
    evals = ['persona', 'sycophancy', 'ai_risk', 'winogenerated', 'global_opinions', 'arc_easy']
    keys = [(i, a) for i in ('bare', 'chat') for a in sorted({r['arm'] for r in diag}, key=ARM_ORDER.index)]
    grid = np.array([[next(r['candidate_mass_pct'] for r in diag if r['interface'] == i and r['arm'] == a and r['eval'] == e)
                      for e in evals] for i, a in keys])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#9ec5f4', '#3987e5', '#104281'])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    for (y, x), v in np.ndenumerate(grid):
        ax.text(x, y, f'{v:.0f}', ha='center', va='center', fontsize=7, color='white' if v > 60 else INK)
    ax.set_xticks(range(len(evals)), [e.replace('_', '\n') for e in evals], fontsize=7)
    ax.set_yticks(range(len(keys)), [f'{i} · {a}' for i, a in keys], fontsize=7); ax.grid(False)
    ax.set_title('Probability mass on the answer options (%)')
    fig.legend(handles=legend_handles(), loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    save(fig, output, 'fig1_format_controls',
         'Mass = total next-token probability on the candidate answers (Yes/No, letters, pronouns). Low mass means the model '
         'mostly wants to say something else; preferences are then read from a small slice of its distribution.')


ARM_ORDER = ['vintage-base', 'vintage-tulu', 'vintage-vsft', 'web-base', 'web-tulu', 'web-vsft']


# ---------------------------------------------------------------- persona
def persona_values(tables, interface):
    v = defaultdict(dict)
    for r in tables['persona']:
        if r['interface'] == interface:
            v[r['behavior']][r['arm']] = r
    return v


def fig_persona_scatter(tables, output, interface, label_n=6):
    plt = setup()
    v = persona_values(tables, interface)
    behaviors = sorted(v)
    get = lambda arm: np.array([v[b][arm]['value'] for b in behaviors])
    vb, vt, wb, wt = get('vintage-base'), get('vintage-tulu'), get('web-base'), get('web-tulu')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    panels = [(vb, wb, 'Vintage base (% matching)', 'Web base (% matching)', 'Pre-SFT: how differently do the bases answer?'),
              (wt - wb, vt - vb, 'Web: Tulu − base (pp)', 'Vintage: Tulu − base (pp)', 'Does Tulu move both bases the same way?'),
              (wb - vb, vt - vb, 'Pre-SFT gap: Web − Vintage (pp)', 'Vintage: Tulu − base (pp)', 'Does Tulu move Vintage toward Web?')]
    for ax, (x, y, xl, yl, title) in zip(axes, panels):
        ax.scatter(x, y, s=12, color=MUTED, alpha=0.55, linewidth=0)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, zorder=0)
        if 'base (%' not in xl:
            ax.axhline(0, color=GRID, linewidth=1); ax.axvline(0, color=GRID, linewidth=1)
        r = np.corrcoef(x, y)[0, 1]
        slope = np.polyfit(x, y, 1)[0]
        ax.text(0.03, 0.97, f'r = {r:.2f} · slope = {slope:.2f} · {len(x)} behaviors', transform=ax.transAxes,
                va='top', fontsize=8, color=INK)
        far = np.argsort(-np.abs(y - x) if 'base (%' in xl else -np.abs(y))[:label_n]
        for i in far:
            ax.annotate(behaviors[i], (x[i], y[i]), fontsize=6, color=INK, xytext=(3, 3), textcoords='offset points')
            ax.scatter([x[i]], [y[i]], s=16, color=INK, linewidth=0)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    save(fig, output, f'fig2_persona_shift_{interface}',
         f'Persona (135 behaviors) · {INTERFACE[interface]} · % = expected probability of the behavior-matching answer, '
         'balanced over Yes/No polarity (50 = no preference).')


def fig_persona_dots(tables, output, interface, categories, name, deltas_only=False):
    plt = setup()
    from .metrics import persona_category
    v = persona_values(tables, interface)
    behaviors = [b for c in categories for b in sorted(v, key=lambda b: -v[b]['web-base']['value']) if persona_category(b) == c]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.22 * len(behaviors) + 1.4), sharey=True)
    for ax, family in zip(axes, ('vintage', 'web')):
        for i, b in enumerate(behaviors):
            base, tulu = v[b][f'{family}-base']['value'], v[b][f'{family}-tulu']['value']
            ax.plot([base, tulu], [i, i], color=COLOR[family], linewidth=1.2, alpha=0.5, zorder=1)
            point(ax, base, i, f'{family}-base', size=22)
            point(ax, tulu, i, f'{family}-tulu', size=22)
            if f'{family}-vsft' in v[b]:
                point(ax, v[b][f'{family}-vsft']['value'], i, f'{family}-vsft', size=14)
        ax.axvline(50, color=MUTED, linewidth=0.8)
        ax.set_xlim(0, 100); ax.set_title(f'{FAMILY[family]}'); ax.set_xlabel('% behavior-matching answer (50 = indifferent)')
        bounds = np.cumsum([sum(persona_category(b) == c for b in behaviors) for c in categories])[:-1]
        for y in bounds:
            ax.axhline(y - 0.5, color=MUTED, linewidth=0.6)
    axes[0].set_yticks(range(len(behaviors)), behaviors, fontsize=7); axes[0].invert_yaxis()
    start = 0
    for c in categories:
        n = sum(persona_category(b) == c for b in behaviors)
        axes[1].text(101, start + n / 2 - 0.5, c, fontsize=8, color=INK, va='center', fontweight='bold')
        start += n
    fig.legend(handles=legend_handles(), loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.0 + 0.6 / (0.22 * len(behaviors) + 1.4)), fontsize=7)
    save(fig, output, f'{name}_{interface}', f'Persona · {INTERFACE[interface]} · lines connect base → Tulu 3 SFT; '
         'squares = Vintage SFT control. Rows sorted by Web base within category.')


def fig_persona_heatmap(tables, output, interface):
    plt = setup()
    from .metrics import PERSONA_CATEGORIES, persona_category
    d = defaultdict(dict)
    for r in tables['persona_deltas']:
        if r['interface'] == interface:
            d[r['behavior']][f"{r['family']}-{r['sft']}"] = r
    categories = list(PERSONA_CATEGORIES) + ['AI self-concept & oversight']
    behaviors = [b for c in categories for b in sorted(d) if persona_category(b) == c]
    cols = [c for c in ('vintage-tulu', 'web-tulu', 'vintage-vsft', 'web-vsft') if c in d[behaviors[0]]]
    grid = np.array([[d[b][c]['delta'] for c in cols] for b in behaviors])
    sig = np.array([[d[b][c]['ci_low'] > 0 or d[b][c]['ci_high'] < 0 for c in cols] for b in behaviors])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', DIVERGING[::-1])
    lim = np.ceil(np.abs(grid).max() / 5) * 5
    fig, ax = plt.subplots(figsize=(5.5, 0.14 * len(behaviors) + 1.5))
    im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
    for (y, x), v in np.ndenumerate(grid):
        ax.text(x, y, f'{v:+.0f}' if sig[y, x] else '', ha='center', va='center', fontsize=5, color=INK)
    ax.set_xticks(range(len(cols)), [c.replace('-tulu', '\nTulu 3').replace('-vsft', '\nVintage SFT') for c in cols], fontsize=7)
    ax.set_yticks(range(len(behaviors)), behaviors, fontsize=5); ax.grid(False)
    start = 0
    for c in categories:
        n = sum(persona_category(b) == c for b in behaviors)
        ax.axhline(start - 0.5, color=SURFACE, linewidth=2)
        ax.text(len(cols) - 0.4, start + n / 2 - 0.5, c, fontsize=6, color=INK, va='center')
        start += n
    fig.colorbar(im, ax=ax, shrink=0.3, label='SFT − base (pp)', pad=0.35)
    ax.set_title(f'Persona: SFT − base, every behavior · {interface}')
    save(fig, output, f'figA_persona_heatmap_{interface}', 'Numbers shown only where the paired 95% bootstrap interval excludes 0.')


# ---------------------------------------------------------------- sycophancy
def fig_sycophancy(tables, output):
    plt = setup()
    rows = [r for r in tables['sycophancy'] if r['measure'] == 'excess_soft']
    lean = tables['political_lean']
    subsets = ['philpapers2020', 'nlp_survey', 'political_typology_quiz']
    titles = {'philpapers2020': 'PhilPapers', 'nlp_survey': 'NLP survey', 'political_typology_quiz': 'Political typology'}
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharey='row')
    for row, interface in enumerate(('bare', 'chat')):
        for col, s in enumerate(subsets):
            ax = axes[row, col]
            sel = sorted([r for r in rows if r['interface'] == interface and r['subset'] == s], key=lambda r: ARM_ORDER.index(r['arm']))
            for i, r in enumerate(sel):
                point(ax, i, r['value'], r['arm'], yerr=ci(r))
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xticks(range(len(sel)), [r['arm'].replace('-', '\n') for r in sel], fontsize=6)
            ax.set_title(f'{titles[s]} · {interface}')
            if col == 0:
                ax.set_ylabel('Agreement with user − bio-blind null (pp)')
        ax = axes[row, 3]
        sel = sorted([r for r in lean if r['interface'] == interface], key=lambda r: ARM_ORDER.index(r['arm']))
        for i, r in enumerate(sel):
            point(ax, i, r['liberal_answer_pct'] - 50, r['arm'], yerr=[[r['liberal_answer_pct'] - r['ci_low']], [r['ci_high'] - r['liberal_answer_pct']]])
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(range(len(sel)), [r['arm'].replace('-', '\n') for r in sel], fontsize=6)
        ax.set_title(f'Own political lean · {interface}'); ax.yaxis.set_tick_params(labelleft=True)
        ax.set_ylabel('Liberal answer − 50 (pp)')
    fig.legend(handles=legend_handles(), loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.03), fontsize=7)
    fig.tight_layout()
    save(fig, output, 'fig4_sycophancy', 'Sycophancy = expected agreement with the view in the user bio minus the agreement expected '
         'if the answer ignored the bio (same question, all bios). Lean = bio-averaged P(liberal answer). 95% cluster-bootstrap over questions.')


# ---------------------------------------------------------------- AI risk
def fig_ai_risk(tables, output, interface, source='human'):
    plt = setup()
    rows = [r for r in tables['ai_risk'] if r['interface'] == interface and r['source'] == source]
    subsets = sorted({r['subset'] for r in rows}, key=lambda s: next(r['value'] for r in rows if r['subset'] == s and r['arm'] == 'web-tulu'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.3 * len(subsets) + 1.4), sharey=True)
    for ax, family in zip(axes, ('vintage', 'web')):
        for i, s in enumerate(subsets):
            get = {r['arm']: r for r in rows if r['subset'] == s}
            base, tulu = get[f'{family}-base'], get[f'{family}-tulu']
            ax.plot([base['value'], tulu['value']], [i, i], color=COLOR[family], linewidth=1.2, alpha=0.5)
            point(ax, base['value'], i, f'{family}-base', size=24, xerr=ci(base))
            point(ax, tulu['value'], i, f'{family}-tulu', size=24, xerr=ci(tulu))
            if f'{family}-vsft' in get:
                point(ax, get[f'{family}-vsft']['value'], i, f'{family}-vsft', size=14)
        ax.axvline(50, color=MUTED, linewidth=0.8); ax.set_xlim(20, 80)
        ax.set_title(FAMILY[family]); ax.set_xlabel('% behavior-matching answer (50 = indifferent)')
    axes[0].set_yticks(range(len(subsets)), [s.split('/')[1] for s in subsets], fontsize=8)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.0 + 0.5 / (0.3 * len(subsets) + 1.4)), fontsize=7)
    save(fig, output, f'fig5_ai_risk_{source}_{interface}', f'Advanced AI risk ({source}-written) · {INTERFACE[interface]} · '
         'expected matching probability averaged over every option order (position bias cancels) · 95% bootstrap.')


# ---------------------------------------------------------------- winogenerated
def fig_winogenerated(tables, output, interface):
    plt = setup()
    occ = [r for r in tables['winogenerated_occupations'] if r['interface'] == interface]
    summ = {r['arm']: r for r in tables['winogenerated'] if r['interface'] == interface}
    arms = [a for a in ARM_ORDER if a in summ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1, 1, 0.9]})
    for arm in arms:
        family, sft = arm_parts(arm)
        ax = axes[0 if family == 'vintage' else 1, ['base', 'tulu', 'vsft'].index(sft)]
        rows = [r for r in occ if r['arm'] == arm]
        x = np.array([r['bls_percent_women'] for r in rows]); y = np.array([100 * r['p_female'] / (r['p_female'] + r['p_male']) for r in rows])
        ax.scatter(x, y, s=9, color=COLOR[family], alpha=0.6, linewidth=0)
        ax.plot([0, 100], [0, 100], color=GRID, linewidth=1, zorder=0)
        s = summ[arm]
        ax.text(0.03, 0.97, f"r = {s['r']:.2f} [{s['r_ci_low']:.2f}, {s['r_ci_high']:.2f}]", transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_title(f'{FAMILY[family]} · {LABEL[sft]}')
        ax.set_xlabel('BLS 2019 % women'); ax.set_ylabel('P(she) / (P(he) + P(she)) (%)')
    for row, family in enumerate(('vintage', 'web')):
        ax = axes[row, 3]
        sel = [a for a in arms if a.startswith(family)]
        for i, a in enumerate(sel):
            s = summ[a]
            point(ax, i, s['neutral_pct'], a, yerr=[[s['neutral_pct'] - s['neutral_ci_low']], [s['neutral_ci_high'] - s['neutral_pct']]])
        ax.set_xticks(range(len(sel)), [LABEL[arm_parts(a)[1]] for a in sel], fontsize=7)
        ax.set_ylim(0, max(5, 1.2 * max(summ[a]['neutral_ci_high'] for a in arms))); ax.set_title(f'{FAMILY[family]} · P(they/their)')
        ax.set_ylabel('Mean P(neutral pronoun) (%)')
    fig.tight_layout()
    save(fig, output, f'fig6_winogenerated_{interface}', f'Winogenerated (299 occupations × 10 sentences) · {INTERFACE[interface]} · '
         'one point per occupation · r = Pearson correlation with BLS 2019 · 95% bootstrap over occupations.')


# ---------------------------------------------------------------- global opinions
def fig_global_opinions(tables, output, interface, min_questions=200, show=12):
    plt = setup()
    rows = [r for r in tables['global_opinions'] if r['interface'] == interface and r['questions'] >= min_questions]
    deltas = [r for r in tables['global_opinions_deltas'] if r['interface'] == interface and r['questions'] >= min_questions]
    sim = defaultdict(dict)
    for r in rows:
        sim[r['country']][r['arm']] = r
    vmw = {r['country']: r for r in deltas if r['family'] == 'vintage-minus-web' and r['sft'] == 'base'}
    vmw_t = {r['country']: r for r in deltas if r['family'] == 'vintage-minus-web' and r['sft'] == 'tulu'}
    order = sorted(vmw, key=lambda c: vmw[c]['delta'])
    picked = order[:show] + order[-show:]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.24 * len(picked) + 1.6), gridspec_kw={'width_ratios': [1.1, 1, 1]})
    ax = axes[0]
    for i, c in enumerate(picked):
        for arm in ('vintage-base', 'web-base'):
            point(ax, sim[c][arm]['value'], i, arm, size=20)
        ax.plot([sim[c]['vintage-base']['value'], sim[c]['web-base']['value']], [i, i], color=MUTED, linewidth=0.8, zorder=1)
        ax.scatter([sim[c]['uniform']['value']], [i], marker='|', color=MUTED, s=30)
    ax.set_yticks(range(len(picked)), [f"{c} ({int(sim[c]['vintage-base']['questions'])})" for c in picked], fontsize=7)
    ax.axhline(show - 0.5, color=MUTED, linewidth=0.6)
    ax.set_xlabel('Similarity to country (1 − JS distance, %)'); ax.set_title('Base models: most Vintage- vs most Web-leaning countries')
    ax = axes[1]
    for i, c in enumerate(picked):
        for d, arm in ((vmw[c], 'vintage-base'), (vmw_t.get(c), 'vintage-tulu')):
            if d:
                point(ax, d['delta'], i + (0.18 if arm.endswith('tulu') else -0.18), arm, size=18, xerr=ci(d, 'delta'))
    ax.axvline(0, color=MUTED, linewidth=0.8); ax.set_yticks(range(len(picked)), [''] * len(picked)); ax.axhline(show - 0.5, color=MUTED, linewidth=0.6)
    ax.set_xlabel('Vintage − Web similarity (pp)'); ax.set_title('Vintage − Web, before (hollow) and after Tulu (filled)')
    ax = axes[2]
    tulu = {(r['family'], r['country']): r for r in deltas if r['sft'] == 'tulu' and r['family'] in ('vintage', 'web')}
    countries = sorted({c for _, c in tulu})
    x = np.array([tulu['web', c]['delta'] for c in countries]); y = np.array([tulu['vintage', c]['delta'] for c in countries])
    ax.scatter(x, y, s=12, color=MUTED, alpha=0.6, linewidth=0)
    for i in np.argsort(-np.abs(y))[:8]:
        ax.annotate(countries[i], (x[i], y[i]), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
    ax.set_xlabel('Web: Tulu − base similarity (pp)'); ax.set_ylabel('Vintage: Tulu − base similarity (pp)')
    ax.set_title(f'Tulu effect per country ({len(countries)} countries)')
    fig.legend(handles=legend_handles()[:2] + legend_handles()[3:4], loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.0 + 0.5 / (0.24 * len(picked) + 1.6)), fontsize=7)
    fig.tight_layout()
    save(fig, output, f'fig7_global_opinions_{interface}', f'GlobalOpinionQA · {INTERFACE[interface]} · countries with ≥{min_questions} '
         'questions · model distribution averaged over original/reversed option order · | = uniform-answer baseline · 95% paired bootstrap over questions.')


def load_tables(root):
    return {p.stem: read(p) for p in Path(root).glob('*.csv')}


def main(root, output):
    tables = load_tables(root)
    output = Path(output)
    fig_format(tables, output)
    for interface in ('bare', 'chat'):
        fig_persona_scatter(tables, output, interface)
        fig_persona_dots(tables, output, interface, ['Personality', 'Politics & social views', 'Religion', 'Ethics'], 'fig3_persona_values')
        fig_persona_dots(tables, output, interface, ['Power, wealth & influence', 'AI self-concept & oversight'], 'figB_persona_ai')
        fig_persona_heatmap(tables, output, interface)
        fig_ai_risk(tables, output, interface)
        fig_ai_risk(tables, output, interface, 'lm')
        fig_winogenerated(tables, output, interface)
        fig_global_opinions(tables, output, interface)
    fig_sycophancy(tables, output)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
