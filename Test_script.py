from d4s import D4S
import matplotlib.pyplot as plt
import numpy as np
import os

COORDS = D4S.read_XYZ('Test_geometry_Li_Bz.xyz')

Functional = "BLYP"
charge=1

params = D4S.get_damping_param(Functional)
S6=params['s6']
S8=params['s8']
a1=params['a1']
a2=params['a2']

for i in COORDS:
    D4_energy, _, _, _ = D4S.D4(i, S6, S8, a1, a2, charge=charge, beta_2=6)
    D4S_energy, _, _, _ = D4S.D4(i, S6, S8, a1, a2, charge=charge, beta_2="D4S")
    D4SL_energy, _, _, _ = D4S.D4(i, S6, S8, a1, a2, charge=charge, beta_2=5, weight_method = "soft_bilinear", smbd=1.0)
    print(f"D4 dispersion correction: {D4_energy}")
    print(f"D4S dispersion correction: {D4S_energy}")
    print(f"D4SL dispersion correction: {D4SL_energy}")