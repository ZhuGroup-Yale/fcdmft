import functools
import os
from dataclasses import dataclass
import dataclasses
import json

import numpy as np

@functools.singledispatch
def gridencoder(x):
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    return x

@gridencoder.register(np.int64)
def _(x):
    return int(x)

@gridencoder.register(np.float64)
def _(x):
    return float(x)

@gridencoder.register(np.ndarray)
def _(x):
    return x.tolist()

@dataclass
class MinimaxGrid:
    """Dataclass for minimax grid data
    """
    n: int
    coefs: np.ndarray
    gridpts: np.ndarray
    err: float
    R: float

    @classmethod
    def from_dict(cls, data):
        return cls(
            n=data['n'],
            coefs=np.array(data['coefs']),
            gridpts=np.array(data['gridpts']),
            err=data['err'],
            R=data['R']
        )


@functools.cache
def load_grids_finite(filename):
    """Load finite grids from a json file

    Parameters
    ----------
    filename : str
        path to the file

    Returns
    -------
    dict
        a dictionary of MinimaxGrid objects, where keys are tuples
        (n, R) representing the number of grid points and the upper
        bound for Emax/Emin
    """
    with open(filename, 'r', encoding='utf-8') as f:
        loadeddata = json.load(f, object_hook=MinimaxGrid.from_dict)
    grids = {}
    for g in loadeddata:
        grids[(g.n, g.R)] = g
    return grids

def search_grids(n, R, grids_finite) -> MinimaxGrid:
    """Find the best grid for a given n and R.

    Parameters
    ----------
    n : int
        number of grid points
    R : float
        upper bound for Emax/Emin
    grids_finite : dict
        dictionary of finite grids, where keys are tuples (n, R)

    Returns
    -------
    MinimaxGrid
        the best matching grid for the given n and R;
        throws ValueError if no suitable grid is found
    """

    assert R > 0, "R must be positive for finite grids"
    keys_correct_n = [k for k in grids_finite.keys() if k[0] == n]
    if not keys_correct_n:
        raise ValueError(f"No grid found for n={n} in finite grids")
    Rs = [k[1] for k in keys_correct_n]
    # return the grid with the smallest R which is still larger
    # than the given R.
    closest_R = min((r for r in Rs if r > R), default=None)
    if closest_R is None:
        raise ValueError(f"No grid found for n={n} and R={R}")
    return grids_finite[(n, closest_R)]


def get_frequency_minimax_grid(k, E_range):
    """Get the frequency minimax grid for a given k and energy range.

    Parameters
    ----------
    k : int
        number of grid points
    E_range : float
        Emax/Emin energy range
    Returns
    -------
    MinimaxGrid or None
        the best matching frequency minimax grid for the given k and energy range.
        Throws ValueError if no suitable grid is found.
    """
    if k != 30:
        raise ValueError("Only k=30 is supported in the current implementation")
    freq_grids_finite_path = os.path.join(os.path.dirname(__file__), "freq_grids_30.json")

    freq_grids_finite = load_grids_finite(freq_grids_finite_path)

    min_avail_R = min((r for (n, r) in freq_grids_finite.keys() if n == k))


    grd = search_grids(k, E_range, freq_grids_finite)

    E_ratio = 1.0
    if E_range < min_avail_R:
        E_ratio = min_avail_R / E_range


    def rescale_grid(aw, E_ratio):
        if E_ratio > 1.5:
            E_ratio = E_ratio / 1.5
        else:
            E_ratio = 1.0
        return aw / E_ratio

    # Note the factor of pi/4 for the weights. This factor only applies to the
    # frequency grid. The grids from CP2K already have this factor absorbed into the weights.
    return (rescale_grid(grd.gridpts, E_ratio), rescale_grid(grd.coefs, E_ratio) * np.pi/4)

def get_time_minimax_grid(k, E_range):
    """Get the time minimax grid for a given k and energy range.

    Parameters
    ----------
    k : int
        number of grid points
    E_range : float
        Emax/Emin energy range

    Returns
    -------
    MinimaxGrid or None
        the best matching frequency time grid for the given k and energy range,
        or None if no suitable grid is found.
    """
    if k != 30:
        raise ValueError("Only k=30 is supported in the current implementation")
    time_grids_finite_path = os.path.join(os.path.dirname(__file__), "time_grids_30.json")
    time_grids_finite = load_grids_finite(time_grids_finite_path)

    min_avail_R = min((r for (n, r) in time_grids_finite.keys() if n == k))

    grd = search_grids(k, E_range, time_grids_finite)

    E_ratio = 1.0
    if E_range < min_avail_R:
        E_ratio = min_avail_R / E_range

    def rescale_grid(aw, E_ratio):
        E_ratio = E_ratio / 1.5 if E_ratio > 1.5 else 1.0
        aw = aw * E_ratio
        return aw

    return (rescale_grid(grd.gridpts, E_ratio), rescale_grid(grd.coefs, E_ratio))
