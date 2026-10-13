"""Seeded policy sampling for stochastic simulation; not a human-behavior model."""
import math


def sample_choice(answer, rng):
    probabilities=answer['probabilities']
    keys=list(probabilities);weights=[probabilities[k] for k in keys]
    if not weights or any(not math.isfinite(w) or w<0 for w in weights) or sum(weights)<=0:
        raise ValueError('Expected finite nonnegative option probabilities')
    return rng.choices(keys,weights=weights,k=1)[0]
