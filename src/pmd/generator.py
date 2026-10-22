"""The eight-token generator and an independent, history-only exact oracle (§3/A1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from functools import lru_cache

import numpy as np

TOKENS = ('^', '0', '1', 'a', 'b', '?', 'R', '$')
ID = {x: i for i, x in enumerate(TOKENS)}
BOS, ZERO, ONE, A, B, QUERY, RESET, END = range(8)
CLUE_POSITION = 17
SPLITS = {'train': (0, 70), 'teacher_val': (70, 75), 'fit': (75, 85),
          'analysis_val': (85, 90), 'test': (90, 100)}


class Stage(IntEnum):
    WAIT = 0
    ANSWER = 1
    END_PENDING = 2
    INVALID = 3
    STOP = 4


def family_tokens(family):
    return [B if (int(family) >> (15-i)) & 1 else A for i in range(16)]


def family_bucket(family):
    text = ''.join(TOKENS[x] for x in family_tokens(family))
    return int(hashlib.sha256(text.encode('ascii')).hexdigest(), 16) % 100


@lru_cache(maxsize=None)
def family_pool(split):
    lo, hi = SPLITS[split]
    return np.array([f for f in range(2**16) if lo <= family_bucket(f) < hi], dtype=np.int64)


def split_manifest():
    return {'hash': 'SHA256 of the 16 literal a/b symbols, interpreted as a big-endian integer mod 100',
            'buckets': SPLITS,
            'pools': {name: dict(count=len(family_pool(name)),
                                sha256=hashlib.sha256(family_pool(name).tobytes()).hexdigest())
                      for name in SPLITS}}


def advance_stage(stage, token):
    stage, token = int(stage), int(token)
    # §3.1's globally absorbing end convention also covers impossible model branches.
    if stage == Stage.STOP or token == END:
        return int(Stage.STOP)
    if stage == Stage.WAIT:
        return int(Stage.ANSWER if token == QUERY else
                   Stage.WAIT if token in (A, B, RESET) else Stage.INVALID)
    if stage == Stage.ANSWER and token in (ZERO, ONE):
        return int(Stage.END_PENDING)
    return int(Stage.INVALID)


@dataclass(frozen=True)
class OracleState:
    belief: Fraction
    stage: int


def oracle_state(history):
    if len(history) < 18:
        raise ValueError('Scientific oracle scoring starts after the clue, at length 18.')
    if history[0] != BOS or any(x not in (A, B) for x in history[1:17]):
        raise ValueError('Not a valid nuisance prefix.')
    clue = history[17]
    if clue not in (ZERO, ONE):
        raise ValueError('Clue must be 0 or 1.')
    belief = Fraction(4, 5) if clue == ONE else Fraction(1, 5)
    stage = int(Stage.WAIT)
    for token in history[18:]:
        if stage == Stage.WAIT and token == RESET:
            belief = Fraction(1, 2)
        elif stage == Stage.ANSWER and token in (ZERO, ONE):
            belief = Fraction(token == ONE)
        stage = advance_stage(stage, token)
    return OracleState(belief, stage)


def oracle_distribution(history):
    state = oracle_state(history)
    p = [Fraction(0)] * 8
    if state.stage == Stage.WAIT:
        for x, mass in [(A, Fraction(2, 5)), (B, Fraction(2, 5)),
                        (QUERY, Fraction(1, 10)), (RESET, Fraction(1, 10))]:
            p[x] = mass
    elif state.stage == Stage.ANSWER:
        p[ONE], p[ZERO] = state.belief, 1-state.belief
    elif state.stage in (Stage.END_PENDING, Stage.STOP):
        p[END] = Fraction(1)
    else:
        raise ValueError('The true generator assigns zero probability to this history.')
    return p


def sample_from_prefix(rng, history, count, reliability=0.8):
    """Sample supported observations, never expose a sampled latent bit to the oracle."""
    out, current = [], list(history)
    for _ in range(count):
        p = np.array([float(x) for x in oracle_distribution(current)])
        x = int(rng.choice(8, p=p))
        out.append(x)
        current.append(x)
        if x == END:
            break
    return out


def episode(rng, family=None, split='train', max_context=64, reliability=0.8):
    if family is None:
        family = int(rng.choice(family_pool(split)))
    hidden = int(rng.integers(2))
    clue = hidden if rng.random() < reliability else 1-hidden
    out = [BOS] + family_tokens(family) + [ID[str(clue)]]
    while len(out) < max_context:
        x = int(rng.choice([A, B, QUERY, RESET], p=[.4, .4, .1, .1]))
        out.append(x)
        if x == RESET:
            hidden = int(rng.integers(2))
        elif x == QUERY:
            return (out + [ID[str(hidden)], END])[:max_context]
    return out  # Right-censored. Never invent a query/end at the boundary.


def make_root(family, clue, waiting, reset=False):
    root = [BOS] + family_tokens(family) + [ID[str(int(clue))]] + list(waiting)
    return root + [RESET] if reset else root


def pad_histories(histories, context=None):
    n = context or max(map(len, histories))
    tokens = np.full((len(histories), n), BOS, dtype=np.int64)
    mask = np.zeros_like(tokens, dtype=bool)
    for i, h in enumerate(histories):
        if len(h) > n:
            raise ValueError('History exceeds context.')
        tokens[i, :len(h)] = h
        mask[i, :len(h)] = True
    return tokens, mask


def fraction_tree(history, horizon):
    if horizon == 0:
        return {(): Fraction(1)}
    out = {}
    for x, p in enumerate(oracle_distribution(history)):
        if p:
            for suffix, mass in fraction_tree(list(history)+[x], horizon-1).items():
                out[(x,)+suffix] = p*mass
    return out


def exact_tv(p, q):
    return sum(abs(p.get(k, 0)-q.get(k, 0)) for k in p.keys() | q.keys()) / 2


def verify_generator():
    nuisance = [A, B] * 8
    h0, h1 = ([BOS]+nuisance+[ZERO]+[A,B,A,B],
              [BOS]+nuisance+[ONE]+[A,B,A,B])
    distances = [exact_tv(fraction_tree(h0, k), fraction_tree(h1, k)) for k in (1, 2, 3)]
    assert distances == [Fraction(0), Fraction(3, 50), Fraction(27, 250)]
    assert oracle_distribution(h0+[QUERY])[ONE] == Fraction(1, 5)
    assert oracle_distribution(h1+[QUERY])[ONE] == Fraction(4, 5)
    for k in (1, 2, 3):
        assert sum(fraction_tree(h0, k).values()) == 1
        assert fraction_tree(h0+[RESET], k) == fraction_tree(h1+[RESET], k)
    assert oracle_state(h0+[RESET]) == oracle_state(h1+[RESET])
    return {'passed': True, 'horizon_tv_exact': [str(x) for x in distances],
            'horizon_tv': [float(x) for x in distances], 'numerical_oracle_error': 0.0,
            'reset_all_future_argument': 'Both histories have identical (belief=1/2, WAIT); the same transition rules apply recursively.',
            'worked_histories': [''.join(TOKENS[x] for x in h) for h in (h0,h1)]}
