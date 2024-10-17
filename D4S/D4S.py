import os
import pickle

import numpy as np
from dftd4.interface import DispersionModel


def read_sec_fortran_data(file_path):
    """Initialize SEC data structures based on the description. Sec datapoint
    are used in the eq 5 (alpha_X_l) to transform REF data.

    Args:
        file_path (str): Filepath of the SEC datafile

    Returns:
        dict: Sorted SEC data.
    """
    sec_data = {
        "secq": np.zeros(17),
        "sscale": np.zeros(17),
        "seccnD3": np.zeros(17),
        "seccn": np.zeros(17),
        "secaiw": np.zeros((23, 17)),
    }
    with open(file_path, "r") as file:
        lines = file.readlines()
    for d, line in enumerate(lines[:-1]):
        if "! SEC" in line:
            for i in range(2):
                data_line_index = i + d + 1
                _, data_name1, ref_n, _, value1, _, _, data_name2, _, \
                    value2 = lines[data_line_index].split()
                ref_n = int(ref_n[1:-1]) - 1
                value1 = float(value1.replace(r"_wp", ""))
                value2 = float(value2.replace(r"_wp/", ""))
                sec_data[data_name1][ref_n] = value1
                sec_data[data_name2][ref_n] = value2
            alpha = []
            for i in range(6):
                data_line_index = i + d + 4
                data_line = lines[data_line_index].split()
                for entry in data_line:
                    if "_wp" in entry:
                        alpha.append(float(entry.replace("_wp", "")
                                           .replace(",", "")))
            sec_data["secaiw"][:, ref_n] = alpha
    return sec_data


def read_fortran_data(file_path):
    """Initialize REF data structures based on the description

    Args:
        file_path (str): Filepath of the REF file

    Returns:
        dict: REF data
    """
    max_elem = 118

    data = {
        "refn": np.zeros(max_elem, dtype=int),  # number of references
        "refq": np.zeros((7, max_elem)),
        "refh": np.zeros((7, max_elem)),
        "gffq": np.zeros((7, max_elem)),
        "gffh": np.zeros((7, max_elem)),
        "dftq": np.zeros((7, max_elem)),
        "dfth": np.zeros((7, max_elem)),
        "pbcq": np.zeros((7, max_elem)),
        "pbch": np.zeros((7, max_elem)),
        "clsq": np.zeros((7, max_elem)),
        "clsh": np.zeros((7, max_elem)),
        "hcount": np.zeros((7, max_elem)),
        "ascale": np.zeros((7, max_elem)),
        "refcovcn": np.zeros((7, max_elem)),
        "refcn": np.zeros(
            (7, max_elem)
        ),  # coordination number used for gaussian weighting
        "refsys": np.zeros((7, max_elem)),
        "alphaiw": np.zeros((23, 7, max_elem)),  # polarizabilities
    }

    with open(file_path, "r") as file:
        lines = file.readlines()

    for d, line in enumerate(lines[:-1]):
        if "! REF" in line:
            if "! REF" in lines[d + 23]:
                read_refn = False
            else:
                read_refn = True
            for i in range(15):
                data_line_index = i + d + 1
                _, data_name, ref_n, atom_n, _, value, _ \
                    = lines[data_line_index].split()
                ref_n = int(ref_n[1:-1]) - 1
                atom_n = int(atom_n[:-1]) - 1
                value = float(value.replace("_wp", ""))
                data[data_name][ref_n][atom_n] = value
            alpha = []
            for i in range(6):
                data_line_index = i + d + 17
                data_line = lines[data_line_index].split()
                for entry in data_line:
                    if "_wp" in entry:
                        alpha.append(float(entry.replace("_wp", "")
                                           .replace(",", "")))
            data["alphaiw"][:, ref_n, atom_n] = alpha

            if read_refn:
                data_line_index = d + 23
                value = lines[data_line_index].split()[3]
                value = int(value)
                data["refn"][atom_n] = int(value)
    return data


def determine_refCNs(atomic_number, data):
    """Determines reference CNs (see Eq. 8 of the original D4 paper) for
    a given atom.

    Args:
        atomic_number (int): Nuclear charge/atomic number of the relevant atom
        data (dict): REF data

    Returns:
        np.ndarray: Reference CNs
    """
    max_cn = 19
    cnc = np.zeros(max_cn)
    cnc[0] = 1
    ref = data["refn"][atomic_number]
    ngw = np.ones(ref)
    for ir in range(ref):
        icn = np.min((round(data["refcovcn"][ir][atomic_number]), max_cn))
        cnc[icn] = cnc[icn] + 1
    for ir in range(ref):
        icn = cnc[np.min((round(data["refcovcn"][ir][atomic_number]), max_cn))]
        ngw[ir] = icn * (icn + 1) / 2
    return ngw


def calculate_ref_C6(atomic_number_1, ref1, atomic_number_2, ref2, data):
    """Calculates reference C6_AB. Reference C6_AB could be seen from the
    expansion of the sum described in eq 9 of original D4 paper.

    Args:
        atomic_number_1 (int): Atomic number of the first atom
        ref1 (int): Ref number of atom 1
        atomic_number_2 (int): Atomic number of the second atom
        ref2 (int): Ref number of atom 2
        data (dict): REF data

    Returns:
        float: Pairwise C6 coefficient
    """
    freq = np.array([0.000001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                     0.8, 0.9, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.5, 3., 4.,
                     5., 7.5, 10.])
    weights = 0.5 * (np.roll(freq, -1) - np.roll(freq, 1))
    weights[0] = 0.5 * (freq[1] - freq[0])
    weights[-1] = 0.5 * (freq[-1] - freq[-2])
    thopi = 3.0 / np.pi
    alpha_product = (
        data["alphaiw"][:, ref1, atomic_number_1]
        * data["alphaiw"][:, ref2, atomic_number_2]
    )
    c6 = thopi * np.sum(alpha_product * weights)
    return c6


def weight(cn, atomic_number, ref_number, beta_2, data,
           weight_method="gaussian"):
    """Calculates the Weight of an individual C6 point for interpolation

    Args:
        cn (float): Coordination number
        atomic_number (int): Atomic number
        ref_number (int): Reference number
        beta_2 (float): Variance parameter
        data (dict): REF data
        weight_method (str, optional): Interpolation method.
        Defaults to "gaussian".

    Returns:
        float: weight of reference C6
    """
    # Keep wrapper method for later implementations
    if weight_method == "gaussian":
        return gaussian_weight(cn, atomic_number, ref_number, beta_2, data)
    return 0.0


def gaussian_weight(cn, atomic_number, ref_number, beta_2, data):
    """Calculates gaussian weights from eq 8 for a particular reference

    Args:
        cn (float): Coordination number
        atomic_number (int): Atomic Number
        ref_number (int): Ref number
        beta_2 (float): variance parameter
        data (dict): REF data

    Returns:
        float: Gaussian weight for reference C6
    """
    N_A_ref = data["refn"][atomic_number]
    Ns = data["Ns_values"][:N_A_ref, atomic_number]
    ref_cn = data["refcovcn"][ref_number][atomic_number]
    numerator = 0
    for j in range(1, int(Ns[ref_number] + 1)):
        numerator += np.exp(-beta_2 * j * (cn - ref_cn) ** 2)
    denominator = 0
    for A_ref in range(N_A_ref):
        ref_cn = data["refcovcn"][A_ref][atomic_number]
        for j in range(1, int(Ns[A_ref] + 1)):
            denominator += np.exp(-beta_2 * j * (cn - ref_cn) ** 2)
    return numerator / denominator


def zeta(q_A, atomic_number, ref_number, data):
    """Zeta-Function (Eq. 2 of original D4 paper) for effective charge

    Args:
        q_A (float): effective charge
        atomic_number (int): Atomic Number
        ref_number (int): Ref number
        data (dict): REF data

    Returns:
        float: evaluated zeta function
    """
    ga = 3  # Charge scaling height (beta_1)
    gc = 2  # Charge scaling steepness
    z_A_ref = (data["clsq"][ref_number][atomic_number]
               + data["Zeff"][atomic_number])
    z_A = q_A + data["Zeff"][atomic_number]
    if z_A < 0:
        zeta = np.exp(ga)
    else:
        z = gc * data["chemical_hardness"][atomic_number] * (1 - z_A_ref / z_A)
        zeta = np.exp(ga * (1 - np.exp(z)))
    return zeta


def get_classical_charges(numbers, positions, charge=0):
    """Evaluate Classical Charges

    Args:
        numbers (ndarray): Atomic numbers of all atoms
        positions (ndarray): XYZ geometry in Bohr
        charge (int, optional): Total charge. Defaults to 0.

    Returns:
        ndarray: Classical Charges
    """
    model = DispersionModel(numbers, positions, charge=charge)
    res = model.get_properties()
    return res["partial charges"]


def get_coordination_numbers(numbers, positions, charge=0):
    """Get Standard D4 coordination numbers

    Args:
        numbers (ndarray): Atomic numbers of all atoms
        positions (ndarray): XYZ geometry in Bohr
        charge (int, optional): Total charge. Defaults to 0.

    Returns:
        ndarray: Coordination Numbers
    """
    model = DispersionModel(numbers, positions, charge=charge)
    res = model.get_properties()
    return res["coordination numbers"]


def get_grimmes_D4_C6(numbers, positions, charge=0):
    """Get Standard D4 C6 coefficients.

    Args:
        numbers (ndarray): Atomic numbers of all atoms
        positions (ndarray): XYZ geometry in Bohr
        charge (int, optional): Total charge. Defaults to 0.

    Returns:
        ndarray: C6 coefficients
    """
    model = DispersionModel(numbers, positions, charge=charge)
    res = model.get_properties()
    return res["c6 coefficients"]


def get_weight_factors(atomic_number, CN_A, beta_2, data):
    """D4SL weight factors.

    Args:
        atomic_number (int): Atomic Number
        CN_A (float): Coordination number
        beta_2 (float): Variance parameter
        data (dict): REF data

    Returns:
        ndarray: D4SL weights
    """
    n_window = 64
    # Window extent: 2 sigma
    x_window = 2.0 / np.sqrt(beta_2)
    n_refdata = data["refn"][atomic_number]
    linear_window = np.linspace(-x_window, x_window, num=n_window + 1)
    gaussian_window = np.exp(-beta_2 * linear_window * linear_window)
    gaussian_window /= np.sum(gaussian_window)
    CN_A_ref = np.array(
        [data["refcovcn"][ref][atomic_number] for ref in range(n_refdata)]
    )

    def weight_factors(cn):
        weights = np.zeros(CN_A_ref.shape, dtype=float)
        if cn <= np.min(CN_A_ref):
            weights[np.argmin(CN_A_ref)] = 1
            return weights
        if cn >= np.max(CN_A_ref):
            weights[np.argmax(CN_A_ref)] = 1
            return weights

        # Find the left neighbour, i. e. biggest value smaller than cn
        mask_left = CN_A_ref < cn
        left_cn = np.max(CN_A_ref[mask_left])
        index_left = np.where(CN_A_ref == left_cn)[0][0]

        # Find the right neighbour, i. e. smallest value bigger than cn
        mask_right = CN_A_ref > cn
        right_cn = np.min(CN_A_ref[mask_right])
        index_right = np.where(CN_A_ref == right_cn)[0][0]

        # Number between 0 and 1 that encodes where cn is in the interval
        cn_int = (cn - left_cn) / (right_cn - left_cn)

        weights[index_right] = cn_int
        weights[index_left] = 1 - cn_int

        return weights

    total_weights = np.zeros(CN_A_ref.shape, dtype=float)

    for i, cn_i in enumerate(linear_window):
        gweight = weight_factors(cn_i + CN_A)
        total_weights += gweight * gaussian_window[i]

    return total_weights


def calculate_C6(atomic_number_A, CN_A, q_A, atomic_number_B, CN_B, q_B,
                 beta_2, data, weight_method="gaussian"):
    """C6 coefficient of a given atomic pair A, B.
       Concurrent evaluation is also possible

    Args:
        atomic_number_A (int): Atomic Number of atom A
        CN_A (float): Coordination Number of atom A
        q_A (float): Charge of atom A
        atomic_number_B (int): Atomic Number of atom B
        CN_B (float): Coordination Number of atom B
        q_B (float): Charge of atom B
        beta_2 (float): Variance parameter
        data (dict): REF data
        weight_method (str, optional): D4SL (soft_bilinear) or D4S (gaussian).
        Defaults to "gaussian".

    Returns:
        float: C6 coefficient
    """
    # Escape concurrent evaluation of many datapoints
    if isinstance(CN_A, tuple):
        return tuple(calculate_C6(atomic_number_A, CN, q_A, atomic_number_B,
                                  CN_B, q_B, beta_2, data, weight_method)
                     for CN in CN_A)
    elif isinstance(CN_A, list):
        return [calculate_C6(atomic_number_A, CN, q_A, atomic_number_B, CN_B,
                             q_B, beta_2, data, weight_method)
                for CN in CN_A]
    elif isinstance(CN_A, np.ndarray):
        return np.array([calculate_C6(atomic_number_A, CN, q_A,
                                      atomic_number_B, CN_B, q_B, beta_2,
                                      data, weight_method) for CN in CN_A])

    # Computes C6_AB coefficient
    N_A_ref = data["refn"][atomic_number_A]
    N_B_ref = data["refn"][atomic_number_B]
    C6 = 0
    # Precompute weights to avoid double computation
    if weight_method == "gaussian":
        weights_A = [
            weight(CN_A, atomic_number_A, ref_i, beta_2, data, weight_method)
            for ref_i in range(N_A_ref)
        ]
        weights_B = [
            weight(CN_B, atomic_number_B, ref_j, beta_2, data, weight_method)
            for ref_j in range(N_B_ref)
        ]

    if weight_method == "gaussian":
        for ref_i in range(N_A_ref):
            zeta_A = zeta(q_A, atomic_number_A, ref_i, data)
            for ref_j in range(N_B_ref):
                zeta_B = zeta(q_B, atomic_number_B, ref_j, data)
                ref_c6 = calculate_ref_C6(
                    atomic_number_A, ref_i, atomic_number_B, ref_j, data
                )
                C6 += (weights_A[ref_i] * weights_B[ref_j]
                       * zeta_A * zeta_B * ref_c6)

    elif weight_method == "soft_bilinear":
        weight_factors_A = get_weight_factors(atomic_number_A, CN_A,
                                              beta_2, data)
        weight_factors_B = get_weight_factors(atomic_number_B, CN_B,
                                              beta_2, data)

        for ref_i in range(N_A_ref):
            zeta_A = zeta(q_A, atomic_number_A, ref_i, data)
            for ref_j in range(N_B_ref):
                ref_c6 = calculate_ref_C6(
                    atomic_number_A, ref_i, atomic_number_B, ref_j, data
                )
                zeta_B = zeta(q_B, atomic_number_B, ref_j, data)
                C6 += (
                    weight_factors_A[ref_i] * weight_factors_B[ref_j]
                    * zeta_A * zeta_B * ref_c6
                )
    return C6


def find_cos_product(a, b, c):
    """Cosine Product required for three-body term

    Args:
        a (float): Number
        b (float): Number
        c (float): Number

    Returns:
        float: cosine product term
    """
    A = a**2 + b**2 - c**2
    B = b**2 + c**2 - a**2
    C = c**2 + a**2 - b**2
    return A * B * C / (8 * a**2 * b**2 * c**2)


def calculate_C8(C6_AB, i, j):
    """Calculate C8 coefficient from C6 coefficient

    Args:
        C6_AB (float): C6 coefficient
        i (float): Index of atom 1
        j (float): Index of atom 2

    Returns:
        float: C8 cpefficient
    """
    return 3 * C6_AB * np.sqrt(np.sqrt(i + 1) * np.sqrt(j + 1)
                               * R4R2[i] * R4R2[j] / 4)


def BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6, C8, N):
    """Becke-Johnson Damping

    Args:
        ATOM_1 (ndarray): charge, position of atom 1
        ATOM_2 (ndarray): charge, position of atom 2
        ALPHA_1 (float): Alpha-Value for atom 1
        ALPHA_2 (float): Alpha-Value for atom 2
        C6 (float): pairwise C6 coefficient
        C8 (float): pairwise C8 coefficient
        N (float): damping power

    Returns:
        float: damping
    """
    R_IJ_cutoff = np.sqrt(C8 / C6)
    R_IJ = np.linalg.norm(ATOM_1[1:] - ATOM_2[1:]) / AUTOANG
    return R_IJ**N / (R_IJ**N + (ALPHA_1 * R_IJ_cutoff + ALPHA_2) ** N)


def TBDamp(R_damp):
    """TB-Damping

    Args:
        R_damp (float): Damping radius

    Returns:
        float: damping
    """
    return 1 / (1 + 6 * R_damp ** (-16))


def read_XYZ(file_name):
    """Read XYZ file in Angstrom

    Args:
        file_name (str): Path to file

    Returns:
        ndarray: charges and coordinates
    """
    # REFERENCE DATA READING PREPARATION AND CURATION
    Periodic_Table = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Th": 90,
        "Pa": 91,
        "U": 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Cn": 112,
        "Nh": 113,
        "Fl": 114,
        "Mc": 115,
        "Lv": 116,
        "Ts": 117,
        "Og": 118,
    }
    with open(file_name, "r") as f:
        file = f.read().split("\n")
    clean_file = []
    for line in file:
        if line != ">" and line != "":
            clean_file.append(line)
    LEN = len(clean_file)
    NUM_STRUCT = LEN // (int(clean_file[0]) + 2)
    COORDS = []
    for i in range(NUM_STRUCT):
        structure = []
        for atoms in range(int(clean_file[0])):
            ind, x, y, z = clean_file[atoms + 2 + i *
                                      (int(clean_file[0]) + 2)].split()
            ind = Periodic_Table[ind]
            x, y, z = float(x), float(y), float(z)
            structure.append([ind, x, y, z])
        COORDS.append(structure)
    return np.array(COORDS)


# Get the directory of the D4S.py
current_dir = os.path.dirname(os.path.abspath(__file__))

AUTOANG = 0.529177249
max_elem = 118
beta_1 = 3
gc = 2
reference_path = os.path.join(current_dir, "reference.inc")
sec_data = read_sec_fortran_data(reference_path)
sec_data["sec_atoms"] = np.array([0, 0, 0, 0, 0, 5, 6, 7, 7, 6,
                                  8, 0, 0, 0, 0, 0, 16])
data = read_fortran_data(reference_path)
data["Ns_values"] = np.zeros((7, max_elem))
for atom in range(max_elem):
    ref = data["refn"][atom]
    data["Ns_values"][:ref, atom] = determine_refCNs(atom, data)

r4r2_path = os.path.join(current_dir, "R2R4.txt")
R4R2 = np.genfromtxt(r4r2_path, delimiter=",")
R4R2 = R4R2[~np.isnan(R4R2)]

data["chemical_hardness"] = np.array(
    [
        0.47259288,
        0.92203391,
        0.17452888,
        0.25700733,
        0.33949086,
        0.42195412,
        0.50438193,
        0.58691863,
        0.66931351,
        0.75191607,
        0.17964105,
        0.22157276,
        0.26348578,
        0.30539645,
        0.34734014,
        0.38924725,
        0.43115670,
        0.47308269,
        0.17105469,
        0.20276244,
        0.21007322,
        0.21739647,
        0.22471039,
        0.23201501,
        0.23933969,
        0.24665638,
        0.25398255,
        0.26128863,
        0.26859476,
        0.27592565,
        0.30762999,
        0.33931580,
        0.37235985,
        0.40273549,
        0.43445776,
        0.46611708,
        0.15585079,
        0.18649324,
        0.19356210,
        0.20063311,
        0.20770522,
        0.21477254,
        0.22184614,
        0.22891872,
        0.23598621,
        0.24305612,
        0.25013018,
        0.25719937,
        0.28784780,
        0.31848673,
        0.34912431,
        0.37976593,
        0.41040808,
        0.44105777,
        0.05019332,
        0.06762570,
        0.08504445,
        0.10247736,
        0.11991105,
        0.13732772,
        0.15476297,
        0.17218265,
        0.18961288,
        0.20704760,
        0.22446752,
        0.24189645,
        0.25932503,
        0.27676094,
        0.29418231,
        0.31159587,
        0.32902274,
        0.34592298,
        0.36388048,
        0.38130586,
        0.39877476,
        0.41614298,
        0.43364510,
        0.45104014,
        0.46848986,
        0.48584550,
        0.12526730,
        0.14268677,
        0.16011615,
        0.17755889,
        0.19497557,
        0.21240778,
        0.07263525,
        0.09422158,
        0.09920295,
        0.10418621,
        0.14235633,
        0.16394294,
        0.18551941,
        0.22370139,
        0.25110000,
        0.25030000,
        0.28840000,
        0.31000000,
        0.33160000,
        0.35320000,
        0.36820000,
        0.39630000,
        0.40140000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
    ]
)
data["Zeff"] = np.array(
    [
        1,
        2,  # H-He
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,  # Li-Ne
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,  # Na-Ar
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,  # K-Kr
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,  # Rb-Xe
        9,
        10,
        11,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,  # Cs-Lu
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,  # Hf-Rn
        9,
        10,
        11,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,  # Fr-Lr
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
    ]
)  # Rf-Og

# COMPUTE ACTUAL ALPHAS

for atomic_number in range(max_elem):
    N_ref = data["refn"][atomic_number]
    for ref_i in range(N_ref):
        _is = int(data["refsys"][ref_i][atomic_number]) - 1
        iz = data["Zeff"][_is]
        t0 = (data["chemical_hardness"][_is] * gc
              * (1 - iz / (iz + data["clsh"][ref_i][atomic_number])))
        aiw = (
            sec_data["sscale"][_is] * sec_data["secaiw"][:, _is]
            * (np.exp(beta_1 * (1 - np.exp(t0))))
        )
        alpha = np.max((data["ascale"][ref_i][atomic_number] *
                        (data["alphaiw"][:, ref_i, atomic_number] -
                         data["hcount"][ref_i][atomic_number] * aiw),
                        np.zeros(23)), axis=0)
        data["alphaiw"][:, ref_i, atomic_number] = alpha

with open(os.path.join(current_dir, "beta_2_m2_300.data"), "rb") as f:
    # optimized with criteria mu_2>-300;
    Beta_2_list_300 = pickle.load(f)


def D4(COORDS, S6, S8, ALPHA_1, ALPHA_2, charge=0, beta_2=6, data=data,
       smbd=1.0, weight_method="gaussian"):
    """D4, D4S or D4SL dispersion energy

    Args:
        COORDS (ndarray): charge and position array as generated by readXYZ
        S6 (float): R^6 coefficient (s6 in Eq. 18 of the original D4 paper)
        S8 (float): R^8 coefficient (s8 in Eq. 18 of the original D4 paper)
        ALPHA_1 (float): Becke-Johnson damping parameter
        ALPHA_2 (float): Becke-Johnson damping parameter
        charge (int, optional): total charge. Defaults to 0.
        beta_2 (int, optional): variance parameter. Defaults to 6.
        data (dict, optional): _description_. Defaults to data.
        smbd (float, optional): _description_. Defaults to 1.0.
        weight_method (str, optional): D4/D4S (gaussian) or
        D4SL (soft bilinear). Defaults to "gaussian".

    Returns:
        float: dispersion energy
    """
    Dispersion_energy_E6 = 0
    Dispersion_energy_E8 = 0
    Dispersion_energy_E_ABC = 0
    CN_array = get_coordination_numbers(
        COORDS[:, 0], COORDS[:, 1:] / AUTOANG, charge=charge
    )
    CHARGE_array = get_classical_charges(
        COORDS[:, 0], COORDS[:, 1:] / AUTOANG, charge=charge
    )
    C6_list = np.zeros((len(COORDS), len(COORDS)))
    C8_list = np.zeros((len(COORDS), len(COORDS)))
    for d1, ATOM_1 in enumerate(COORDS[:-1]):
        for d2, ATOM_2 in enumerate(COORDS[d1 + 1:]):
            R_IJ = (
                np.linalg.norm(ATOM_1[1:] - ATOM_2[1:]) / AUTOANG
            )  # Divide by the transformation from A to bohr
            atomic_number_A = int(ATOM_1[0] - 1)
            atomic_number_B = int(ATOM_2[0] - 1)
            if isinstance(beta_2, str):
                if beta_2 == "D4S":
                    beta_2 = Beta_2_list_300[atomic_number_A][atomic_number_B]
            C6_AB = calculate_C6(
                atomic_number_A, CN_array[d1], CHARGE_array[d1],
                atomic_number_B, CN_array[d2 + d1 + 1],
                CHARGE_array[d2 + d1 + 1], beta_2, data,
                weight_method=weight_method
            )
            C6_list[d1][d2 + d1 + 1] = calculate_C6(
                atomic_number_A, CN_array[d1], 0,
                atomic_number_B, CN_array[d2 + d1 + 1], 0,
                beta_2, data, weight_method=weight_method
            )
            C8_AB = calculate_C8(C6_AB, atomic_number_A, atomic_number_B)
            C8_list[d1][d2 + d1 + 1] = calculate_C8(C6_list[d1][d2 + d1 + 1],
                                                    atomic_number_A,
                                                    atomic_number_B)
            Dispersion_energy_E6 += (
                S6 * BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6_AB, C8_AB, 6)
                * C6_AB / R_IJ**6
            )
            Dispersion_energy_E8 += (
                S8 * BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6_AB, C8_AB, 8)
                * C8_AB / R_IJ**8
            )
    C6_list += C6_list.T
    C8_list += C8_list.T
    for d1, ATOM_1 in enumerate(COORDS[:-2]):
        for d2, ATOM_2 in enumerate(COORDS[d1 + 1: -1]):
            for d3, ATOM_3 in enumerate(COORDS[d1 + d2 + 2:]):
                rij = np.linalg.norm(ATOM_1[1:] - ATOM_2[1:]) / AUTOANG
                rjk = np.linalg.norm(ATOM_2[1:] - ATOM_3[1:]) / AUTOANG
                rki = np.linalg.norm(ATOM_3[1:] - ATOM_1[1:]) / AUTOANG
                cos_factor = find_cos_product(rij, rjk, rki)
                C9 = np.sqrt(
                    C6_list[d1][d1 + d2 + 1]
                    * C6_list[d1 + d2 + 1][d1 + d2 + d3 + 2]
                    * C6_list[d1 + d2 + d3 + 2][d1]
                )
                R_damp_ij = (
                    ALPHA_1 * np.sqrt(C8_list[d1][d1 + d2 + 1]
                                      / C6_list[d1][d1 + d2 + 1]) + ALPHA_2
                )
                R_damp_jk = (
                    ALPHA_1 * np.sqrt(
                        C8_list[d1 + d2 + 1][d1 + d2 + d3 + 2]
                        / C6_list[d1 + d2 + 1][d1 + d2 + d3 + 2]
                    ) + ALPHA_2
                )
                R_damp_ki = (
                    ALPHA_1
                    * np.sqrt(
                        C8_list[d1 + d2 + d3 + 2][d1] /
                        C6_list[d1 + d2 + d3 + 2][d1]
                    ) + ALPHA_2
                )
                R_damp = ((rij * rjk * rki) /
                          (R_damp_ij * R_damp_jk * R_damp_ki)) ** (1 / 3)
                Dispersion_energy_E_ABC += (
                    smbd * TBDamp(R_damp) * C9 * (3 * cos_factor + 1)
                    / (rij * rjk * rki) ** 3
                )
    total_disp_energy = -(Dispersion_energy_E8 + Dispersion_energy_E6 -
                          Dispersion_energy_E_ABC)
    return (
        total_disp_energy,
        C6_list,
        Dispersion_energy_E6,
        Dispersion_energy_E8,
    )
