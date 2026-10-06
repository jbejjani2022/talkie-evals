import math
import numpy as np
from behavior_evals import metrics


def lp(*p):
    return {'logprobs': [math.log(x) for x in p]}


def test_persona_yes_bias_is_neutral_and_signal_survives_bias():
    items = [{'match': 0}, {'match': 0}, {'match': 1}, {'match': 1}]  # Yes matches, Yes matches, No matches...
    biased = [lp(0.9, 0.1)] * 4
    c = metrics.persona_choices(items, biased, metrics.persona_offset(biased))
    point, _ = metrics.stratified_mean(c['raw'], c['polarity'], reps=10)
    assert point[0] == 0.5  # always "Yes": half the statements match
    soft, _ = metrics.stratified_mean(c['soft'], c['polarity'], reps=10)
    assert np.isclose(soft[0], 0.5)  # the primary (expected-probability) metric is exactly bias-neutral too
    signal = [lp(0.95, 0.05), lp(0.95, 0.05), lp(0.6, 0.4), lp(0.6, 0.4)]  # Yes-biased but discriminating
    c = metrics.persona_choices(items, signal, metrics.persona_offset(signal))
    point, _ = metrics.stratified_mean(c['calibrated'], c['polarity'], reps=10)
    assert point[0] == 1 and c['raw'].mean() == 0.5


def test_sycophancy_null_removes_letter_bias():
    items = [{'group': 'q', 'match': 0}, {'group': 'q', 'match': 1}]
    always_a = metrics.sycophancy_parts(items, [lp(0.8, 0.2), lp(0.8, 0.2)])
    assert always_a['observed'].mean() == always_a['null'].mean() == 0.5
    follower = metrics.sycophancy_parts(items, [lp(0.8, 0.2), lp(0.2, 0.8)])
    assert follower['observed'].mean() == 1 and follower['null'].mean() == 0.5


def test_ai_risk_rotation_cancels_position_bias():
    items = [{'group': 'g', 'match': 0, 'not_match': [1], 'order': [0, 1]},
             {'group': 'g', 'match': 1, 'not_match': [0], 'order': [1, 0]}]
    _, values = metrics.ai_risk_items(items, [lp(0.7, 0.3), lp(0.7, 0.3)])
    assert values['hard'][0] == 0.5 and values['consistent'][0] == 0


def test_global_opinions_maps_reversed_order_back_to_content():
    items = [{'group': 'g', 'order': [0, 1, 2]}, {'group': 'g', 'order': [2, 1, 0]}]
    d = metrics.global_opinions_distributions(items, [lp(0.6, 0.3, 0.1), lp(0.1, 0.3, 0.6)])
    assert np.allclose(d['g'], [0.6, 0.3, 0.1])
    assert math.isclose(metrics.js_similarity([0.2, 0.8], [0.2, 0.8]), 1) and metrics.js_similarity([1, 0], [0, 1]) == 0


def test_clustered_mean_is_paired_across_arms():
    point, draws = metrics.clustered_mean([[1, 1, 0, 0], [1, 1, 0, 0]], ['a', 'a', 'b', 'b'], reps=50)
    assert np.allclose(point, 0.5) and np.allclose(draws[:, 0], draws[:, 1])
