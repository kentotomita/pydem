"""crater function"""
import numpy as np
from numpy.random import RandomState

from .ds import dsa
from .rock import rocky_terrain

def crater_count(area: float, d: float, dmax: float, dmin: float, rng: RandomState=None) -> int:
    """return number of crater given size

    Args:
        area: field area in m
        d: crater rim diameter in m
        dmax: max diameter in m
        dmin: min diameter in m
    """
    assert dmax ** 2 < area
    area_km = area / 1000 / 1000

    if rng is None:
        rng = RandomState(seed=0)

    num = 0

    if dmin < d < dmax:
        if d < 50:
            expectation = 10 * area_km
            if expectation >= 1:
                num = int(expectation)
            elif rng.rand() < expectation:
                num = 1
            else:
                num = 0
        else:  # dr>=50 m
            expectation = ((50 / d) ** 2) * area_km
            if expectation >= 1:
                num = int(expectation)
            else:
                if np.random.rand() < expectation:
                    num = 1
    return num


def gen_crater(d: float, res: float, rng: RandomState=None):
    """
    Generate crater height map of size by the following steps
    (1) Generate crater base height. The height could be calculated using the distance from the crater center
    Args:
        d: crater rim diameter (m)
        res: map resolution (m/pix)
        rng: random state generator
    """
    if rng is None:
        rng = RandomState(seed=0)

    # span [m]
    span = int(2.4 * d / 2)

    X = np.arange(-span, span, res)
    Y = np.arange(-span, span, res)
    X, Y = np.meshgrid(X, Y)
    r = (X ** 2 + Y ** 2) ** (1 / 2)  # distance from the center
    assert X.shape[0]==X.shape[1]
    n = X.shape[0]

    # (1) Generate the base height of the crater (the base height depends on the distance from the center) ---------
    # rim_crest_diameter [km]
    Dr_km = d * 1e-3
    # Depth [km]
    Ri_km = 0.196 * Dr_km ** 1.010
    # Rim height [km]
    Re_km = 0.036 * Dr_km ** 1.014
    # Width of rim flank [km]
    We_km = 0.257 * Dr_km ** 1.011
    # Floor diameter [km]
    D_Fkm = 0.031 * Dr_km ** 1.765
    # Rimwall width [km]
    Wi_km = (Dr_km - D_Fkm) / 2

    # km --> mR
    Dr = Dr_km * 1e3
    Ri = Ri_km * 1e3
    Re = Re_km * 1e3
    We = We_km * 1e3
    Df = D_Fkm * 1e3
    Wi = Wi_km * 1e3

    # crater base
    # list of radius to determine the crater's shape
    R1 = Df / 2   # radius of Floor
    R2 = Dr / 2   # radius of Rim
    R3 = R2 + We  # radius of rim flank

    # add noise to the list of radius = shape of the crater
    fractal = dsa(rng.rand(n, n))
    R1 = R1 + fractal * R1 * 0.2
    R2 = R2 + fractal * R2 * 0.2
    R3 = R3 + fractal * R3 * 0.2

    # base shape
    Z_base = (r <= R1) * (Re - Ri) \
             + (R1 < r) * (r <= Dr / 2) * (Ri / (R2 - R1) ** 2 * (r - R1) ** 2 + Re - Ri) \
             + (R2 < r) * (r <= R3) * (Re * (R3 - r) / We) \
             + (R3 < r) * 0

    # (2) height of the rock inside the crater --------------------------------------------------------------------
    D_max = 0.02 * Dr  # maximum rock diameter
    k_A = 0.2  # coverage_rate of zone A
    k_B1 = 0.4  # coverage_rate of zone B1
    k_B2 = 0.7  # coverage_rate of zone B2
    k_C = 0.35  # coverage_rate of zone C
    k_D = 0.25  # coverage_rate of zone D
    k_E = 0.20  # coverage_rate of zone E
    shape = (n, n)
    rock_A = rocky_terrain(shape, res, k_A, D_max, dmin=res * 2)
    rock_B1 = rocky_terrain(shape, res, k_B1, D_max, dmin=res * 2)
    rock_B2 = rocky_terrain(shape, res, k_B2, D_max, dmin=res * 2)
    rock_C = rocky_terrain(shape, res, k_C, D_max, dmin=res * 2)
    rock_D = rocky_terrain(shape, res, k_D, D_max, dmin=res * 2)
    rock_E = rocky_terrain(shape, res, k_E, D_max, dmin=res * 2)

    Z_rock = (0 < r) * (r <= 0.3 * Dr / 2) * rock_A \
        + (0.3 * Dr / 2 < r) * (r <= 0.9 * Dr / 2) * rock_B1 \
        + (0.9 * Dr / 2 < r) * (r <= 1.0 * Dr / 2) * rock_B2 \
        + (1.0 * Dr / 2 < r) * (r <= 1.4 * Dr / 2) * rock_C \
        + (1.4 * Dr / 2 < r) * (r <= 2.0 * Dr / 2) * rock_D \
        + (2.0 * Dr / 2 < r) * (r <= 2.4 * Dr / 2) * rock_E

    Z = Z_base + Z_rock

    return Z

