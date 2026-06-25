from typing import Dict

from econcausal.keel import DirectedGraph


def structural_hamming(prediction: DirectedGraph, truth: DirectedGraph) -> int:
    predicted = prediction.edge_set()
    actual = truth.edge_set()
    reversed_pairs = {
        (i, j) for (i, j) in actual if (j, i) in predicted and (i, j) not in predicted
    }
    reversals = len(reversed_pairs)
    missing = len(actual - predicted) - reversals
    extra = len(predicted - actual) - reversals
    return missing + extra + reversals


def edge_scores(prediction: DirectedGraph, truth: DirectedGraph) -> Dict[str, float]:
    predicted = prediction.edge_set()
    actual = truth.edge_set()
    true_positive = len(predicted & actual)
    false_positive = len(predicted - actual)
    false_negative = len(actual - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if actual else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "shd": float(structural_hamming(prediction, truth)),
    }


def interaction_ratio(full: float, text_only: float, stats_only: float, baseline: float) -> float:
    gain_full = full - baseline
    gain_text = text_only - baseline
    gain_stats = stats_only - baseline
    denominator = gain_text + gain_stats
    if denominator == 0.0:
        return 1.0
    return gain_full / denominator
