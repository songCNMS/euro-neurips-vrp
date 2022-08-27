import numpy as np
from environment import State


def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res


def _greedy(observation: State, rng: np.random.Generator):
    return {
        **observation,
        'must_dispatch': np.ones_like(observation['must_dispatch']).astype(np.bool8)
    }


def _lazy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    return _filter_instance(observation, mask)


def _random(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)


def _heuristic(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    duration_matrix = observation["duration_matrix"]
    median_duration = np.median(duration_matrix)
    for i, m in enumerate(mask):
        if m: continue
        min_dist = sorted([(duration_matrix[i][j]+duration_matrix[j][i])/2.0 for j, _m in enumerate(mask) if _m])[0]
        if min_dist <= median_duration: mask[i] = True
    # mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)


STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    heuristic=_heuristic
)
