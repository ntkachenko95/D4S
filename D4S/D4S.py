import dftd4
from dftd4.interface import DampingParam, DispersionModel
import numpy as np
import os

# Get the directory of the D4S.py
current_dir = os.path.dirname(os.path.abspath(__file__))

def read_sec_fortran_data(file_path):
    # Initialize SEC data structures based on the description
    # Sec data point are used in the eq 5 (alpha_X_l) to transform REF data
    max_elem = 118
    
    sec_data = {
      'secq': np.zeros(17),
      'sscale': np.zeros(17),
      'seccnD3': np.zeros(17),
      'seccn': np.zeros(17),
      'secaiw': np.zeros((23,17))}
    with open(file_path, 'r') as file:
        lines = file.readlines()   
    for d, line in enumerate(lines[:-1]):
        if "! SEC" in line:
            for i in range(2):
                data_line_index = i+d+1
                _, data_name1, ref_n, _, value1, _, _, data_name2, _, value2 = lines[data_line_index].split()
                ref_n = int(ref_n[1:-1])-1
                value1 = float(value1.replace('_wp', ''))
                value2 = float(value2.replace(r'_wp/', ''))
                sec_data[data_name1][ref_n] = value1
                sec_data[data_name2][ref_n] = value2
            alpha = []
            for i in range(6):
                data_line_index = i+d+4
                data_line = lines[data_line_index].split()
                #print(lines[data_line_index])
                for entry in data_line:
                    if "_wp" in entry:
                        alpha.append(float(entry.replace('_wp', '').replace(',','')))
            #print(ref_n)
            sec_data['secaiw'][:,ref_n] = alpha   
    return sec_data

def read_fortran_data(file_path):
    # Initialize REF data structures based on the description
    max_elem = 118
    
    data = {
        'refn': np.zeros(max_elem, dtype=np.int64), #numer of references
        'refq': np.zeros((7, max_elem)),
        'refh': np.zeros((7, max_elem)),
        'gffq': np.zeros((7, max_elem)),
        'gffh': np.zeros((7, max_elem)),
        'dftq': np.zeros((7, max_elem)),
        'dfth': np.zeros((7, max_elem)),
        'pbcq': np.zeros((7, max_elem)),
        'pbch': np.zeros((7, max_elem)),
        'clsq': np.zeros((7, max_elem)),
        'clsh': np.zeros((7, max_elem)),
        'hcount': np.zeros((7, max_elem)),
        'ascale': np.zeros((7, max_elem)),
        'refcovcn': np.zeros((7, max_elem)),
        'refcn': np.zeros((7, max_elem)), #coordination number used for gaussian weighting
        'refsys': np.zeros((7, max_elem)),
        'alphaiw': np.zeros((23, 7, max_elem)) #polarisabilities
    }
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for d, line in enumerate(lines[:-1]):
        if "! REF" in line:
            if "! REF" in lines[d+23]:
                read_refn = False
            else:
                read_refn = True
            for i in range(15):
                data_line_index = i+d+1
                #print(lines[data_line_index])
                _, data_name, ref_n, atom_n, _, value, _ = lines[data_line_index].split()
                ref_n = int(ref_n[1:-1])-1
                atom_n = int(atom_n[:-1])-1
                value = float(value.replace('_wp', ''))
                data[data_name][ref_n][atom_n] = value
            alpha = []
            for i in range(6):
                data_line_index = i+d+17
                data_line = lines[data_line_index].split()
                #print(lines[data_line_index])
                for entry in data_line:
                    if "_wp" in entry:
                        alpha.append(float(entry.replace('_wp', '').replace(',','')))
            data['alphaiw'][:,ref_n,atom_n] = alpha

            if read_refn:
                data_line_index = d+23
                _, _, _, value, _ = lines[data_line_index].split()
                value = int(value)
                data['refn'][atom_n] = int(value)                
    return data

    
def determine_Ns(atomic_number, data):
    #Determines Ns number that comes to eq 8
    max_cn = 19
    cnc = np.zeros(max_cn)
    cnc[0] = 1
    ref = data['refn'][atomic_number]
    ngw = np.ones(ref)
    for ir in range(ref):
        icn = np.min((round(data['refcn'][ir][atomic_number]), max_cn))
        cnc[icn] = cnc[icn] + 1
    for ir in range(ref):
        icn = cnc[np.min((round(data['refcn'][ir][atomic_number]), max_cn))]
        ngw[ir] = icn*(icn+1)/2
    return ngw

def calculate_ref_C6(atomic_number_i, ref_i, atomic_number_j, ref_j, data):
    #Calculates reference C6_AB. Reference C6_AB could be seen from the expansion of the sum described in eq 9.
    freq = np.array([0.000001, 0.050000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.200000, 1.400000, 1.600000, 1.800000, 2.000000, 2.500000,
            3.000000, 4.000000, 5.000000, 7.500000, 10.00000])
    weights = 0.5 * np.array([ ( freq[1] - freq[0] ), 
                              (freq[1] - freq[0]) + (freq[2] - freq[1]),
                              (freq[2] - freq[1]) + (freq[3] - freq[2]),
                              (freq[3] - freq[2]) + (freq[4] - freq[3]),
                              (freq[4] - freq[3]) + (freq[5] - freq[4]),
                              (freq[5] - freq[4]) + (freq[6] - freq[5]),
                              (freq[6] - freq[5]) + (freq[7] - freq[6]),
                              (freq[7] - freq[6]) + (freq[8] - freq[7]),
                              (freq[8] - freq[7]) + (freq[9] - freq[8]),
                              (freq[9] - freq[8]) + (freq[10] - freq[9]),
                              (freq[10] - freq[9]) + (freq[11] - freq[10]),
                              (freq[11] - freq[10]) + (freq[12] - freq[11]),
                              (freq[12] - freq[11]) + (freq[13] - freq[12]),
                              (freq[13] - freq[12]) + (freq[14] - freq[13]),
                              (freq[14] - freq[13]) + (freq[15] - freq[14]),
                              (freq[15] - freq[14]) + (freq[16] - freq[15]),
                              (freq[16] - freq[15]) + (freq[17] - freq[16]),
                              (freq[17] - freq[16]) + (freq[18] - freq[17]),
                              (freq[18] - freq[17]) + (freq[19] - freq[18]),
                              (freq[19] - freq[18]) + (freq[20] - freq[19]),
                              (freq[20] - freq[19]) + (freq[21] - freq[20]),
                              (freq[21] - freq[20]) + (freq[22] - freq[21]),
                              (freq[22] - freq[21])])
    thopi = 3.0/np.pi
    alpha_product = data['alphaiw'][:,ref_i,atomic_number_i]*data['alphaiw'][:,ref_j,atomic_number_j]
    c6 = thopi*np.sum(alpha_product*weights)
    return c6
    
def gaussian_weight(CN_A, atomic_number, ref_number, beta_2, data):
    #Calculates gaussian weights from eq 8 for a particular reference
    N_A_ref = data['refn'][atomic_number]
    Ns = data['Ns_values'][:N_A_ref,atomic_number]
    CN_A_ref = data['refcovcn'][ref_number][atomic_number]
    numerator = 0
    for j in range(1,int(Ns[ref_number]+1)):
        numerator += np.exp(-beta_2*j*(CN_A - CN_A_ref)**2)
    denominator = 0
    for A_ref in range(N_A_ref):
        CN_A_ref = data['refcovcn'][A_ref][atomic_number]
        for j in range(1,int(Ns[A_ref]+1)):
            denominator += np.exp(-beta_2*j*(CN_A - CN_A_ref)**2)
    return numerator/denominator
    
def zeta(q_A, atomic_number, ref_number, data):
    #Zeta function from eq 2
    ga = 3 #Charge scaling height (beta_1)
    gc = 2 #Charge scaling steepness
    z_A_ref = data['clsq'][ref_number][atomic_number]+data['Zeff'][atomic_number]
    z_A = q_A+data['Zeff'][atomic_number]
    if z_A<0:
        zeta = np.exp(ga)
    else:
        zeta = np.exp(ga*(1-np.exp(gc*data['chemical_hardness'][atomic_number]*(1-z_A_ref/z_A))))
    return zeta

#Three functions that uses Grimme's D4 library to compute charges, CNs, and C6 coefficients for a given geometry
#Used here for comparison with our implementation 
def get_classical_charges(numbers, positions,method="blyp"):
    model = DispersionModel(numbers, positions)
    res = model.get_properties()
    return res['partial charges']

def get_coordination_numbers(numbers, positions,method="blyp"):
    model = DispersionModel(numbers, positions)
    res = model.get_properties()
    return res['coordination numbers']

def get_grimmes_D4_C6(numbers, positions,method="blyp"):
    model = DispersionModel(numbers, positions)
    res = model.get_properties()
    return res['c6 coefficients']
    
def calculate_c6(atomic_number_A, CN_A, q_A, atomic_number_B, CN_B, q_B, beta_2, data):
    #Computes C6_AB coefficient
    N_A_ref = data['refn'][atomic_number_A]
    N_B_ref = data['refn'][atomic_number_B]
    C6 = 0
    for ref_i in range(N_A_ref):
        for ref_j in range(N_B_ref):
            ref_c6 = calculate_ref_C6(atomic_number_A, ref_i, atomic_number_B, ref_j, data)
            #print(f"REF: {ref_i} (CN={data['refcn'][ref_i][atomic_number_A]}), {ref_j} (CN={data['refcn'][ref_j][atomic_number_B]}). C6: {ref_c6}")
            W_A = gaussian_weight(CN_A, atomic_number_A, ref_i, beta_2, data)
            zetta_A = zeta(q_A, atomic_number_A, ref_i, data)
            W_B = gaussian_weight(CN_B, atomic_number_B, ref_j, beta_2, data)
            zetta_B = zeta(q_B, atomic_number_B, ref_j, data)
            #print(f'W_A*W_B: {W_A*W_B}')
            C6 += W_A*zetta_A*W_B*zetta_B*ref_c6
    return C6

#REFERENCE DATA READING PREPARATION AND CURATION

max_elem = 118
beta_1 = 3
gc = 2
reference_path = os.path.join(current_dir, "reference.inc")
sec_data = read_sec_fortran_data(reference_path)
sec_data['sec_atoms'] = np.array([0, 0, 0, 0, 0, 5, 6, 7, 7, 6, 8, 0, 0, 0, 0, 0, 16])
data = read_fortran_data(reference_path)
data['Ns_values'] = np.zeros((7, max_elem))
for atom in range(max_elem):
    ref = data['refn'][atom]
    data['Ns_values'][:ref,atom] = determine_Ns(atom, data)
data['chemical_hardness'] = np.array([0.47259288, 0.92203391, 0.17452888, 0.25700733, 0.33949086,
                                      0.42195412, 0.50438193, 0.58691863, 0.66931351, 0.75191607,
                                      0.17964105, 0.22157276, 0.26348578, 0.30539645, 0.34734014,
                                      0.38924725, 0.43115670, 0.47308269, 0.17105469, 0.20276244,
                                      0.21007322, 0.21739647, 0.22471039, 0.23201501, 0.23933969,
                                      0.24665638, 0.25398255, 0.26128863, 0.26859476, 0.27592565,
                                      0.30762999, 0.33931580, 0.37235985, 0.40273549, 0.43445776,
                                      0.46611708, 0.15585079, 0.18649324, 0.19356210, 0.20063311,
                                      0.20770522, 0.21477254, 0.22184614, 0.22891872, 0.23598621,
                                      0.24305612, 0.25013018, 0.25719937, 0.28784780, 0.31848673,
                                      0.34912431, 0.37976593, 0.41040808, 0.44105777, 0.05019332,
                                      0.06762570, 0.08504445, 0.10247736, 0.11991105, 0.13732772,
                                      0.15476297, 0.17218265, 0.18961288, 0.20704760, 0.22446752,
                                      0.24189645, 0.25932503, 0.27676094, 0.29418231, 0.31159587,
                                      0.32902274, 0.34592298, 0.36388048, 0.38130586, 0.39877476,
                                      0.41614298, 0.43364510, 0.45104014, 0.46848986, 0.48584550,
                                      0.12526730, 0.14268677, 0.16011615, 0.17755889, 0.19497557,
                                      0.21240778, 0.07263525, 0.09422158, 0.09920295, 0.10418621,
                                      0.14235633, 0.16394294, 0.18551941, 0.22370139, 0.25110000,
                                      0.25030000, 0.28840000, 0.31000000, 0.33160000, 0.35320000,
                                      0.36820000, 0.39630000, 0.40140000, 0.00000000, 0.00000000,
                                      0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                      0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                      0.00000000, 0.00000000, 0.00000000])
data['Zeff'] = np.array([ 1,                                                 2,  # H-He
                          3, 4,                               5, 6, 7, 8, 9,10,  # Li-Ne
                         11,12,                              13,14,15,16,17,18,  # Na-Ar
                         19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,  # K-Kr
                          9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,  # Rb-Xe
                          9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,  # Cs-Lu
                         12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, # Hf-Rn
                          9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,  # Fr-Lr
                         12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]) # Rf-Og

#COMPUTE ACTUAL ALPHAS

for atomic_number in range(max_elem):
    N_ref = data['refn'][atomic_number]
    for ref_i in range(N_ref):
        _is = int(data['refsys'][ref_i][atomic_number])-1
        iz = data['Zeff'][_is]
        aiw = sec_data['sscale'][_is]*sec_data['secaiw'][:,_is]*(np.exp(beta_1*(1-np.exp(data['chemical_hardness'][_is]*gc*(1-iz/(iz+data['clsh'][ref_i][atomic_number]))))))
        alpha = np.max((data['ascale'][ref_i][atomic_number]*(data['alphaiw'][:,ref_i,atomic_number]-data['hcount'][ref_i][atomic_number]*aiw),np.zeros(23)),axis=0)
        data['alphaiw'][:,ref_i,atomic_number] = alpha