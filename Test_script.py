from D4S import D4S
import matplotlib.pyplot as plt
import numpy as np

#Compute C6 coeffs wrt CNs
atoms = [5, 6, 7] #C, N, O
atoms_label = ["C","N","O"]
data = D4S.data

CNs = np.linspace(0,5,1000)
for d, atom in enumerate(atoms):
    Y = D4S.calculate_c6(atom, CNs, 0, atom, CNs, 0, 6, data)
    plt.plot(CNs,Y,label=atoms_label[d])
plt.xlabel("CN")
plt.ylabel(r"$C_6^{AA}$ (a.u.)")
plt.legend()
plt.tight_layout()
plt.savefig("C6_vs_CN.jpg", dpi=600)
