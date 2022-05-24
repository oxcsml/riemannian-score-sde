# %%
%load_ext autoreload
%autoreload 2

# %%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# %%
from Bio import PDB
from Bio.PDB import internal_coords
import nglview as nv
# ic_structure = PDB.internal_coords.IC_Chain(structure)
# ic_structure.internal_to_atom_coordinates()
import glob
datadir = "/data/localhost/not-backed-up/mhutchin/score-sde/data/abdb/raw_antibodies"
files = glob.glob(datadir + '/*.pdb')
# %%
parser = PDB.PDBParser()
structure = parser.get_structure('test_struct', "/data/localhost/not-backed-up/mhutchin/score-sde/data/abdb/raw_antibodies/1A2Y_1.pdb")
structure.atom_to_internal_coordinates()

chain = next(structure.get_chains())
angles = np.zeros((len(list(chain.get_residues())), 3))
lengths = np.zeros((len(list(chain.get_residues())), 3))
for i, r in enumerate(chain.get_residues()):
    if r.internal_coord.rprev:
        angles[i-1, 0] = r.internal_coord.get_angle("omega")
        angles[i-1, 1] = r.internal_coord.get_angle("phi")

    lengths[i, 0] = r.internal_coord.get_length("N:CA")
    lengths[i, 1] = r.internal_coord.get_length("CA:C")

    if r.internal_coord.rnext:
        angles[i, 2] = r.internal_coord.get_angle("psi")
        lengths[i, 2] = r.internal_coord.get_length("C:1N")

# %%
sns.histplot(lengths[:, 0])
sns.histplot(lengths[:, 1])
sns.histplot(lengths[:, 2])

# %%
sns.set_theme(style="dark")


x=angles[:, 1]
y=angles[:,2]
sns.scatterplot(x=x,y=y, s=5, color=".15")
# sns.histplot(x=x, y=y, bins=50, pthresh=-1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)

plt.gca().set_aspect('equal')

# %%
sns.set_theme(style="dark")


x=angles[:,1]
y=angles[:,0]
sns.scatterplot(x=x,y=y, s=5, color=".15")
# sns.histplot(x=x, y=y, bins=50, pthresh=-1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)

plt.gca().set_aspect('equal')
# %%
sns.set_theme(style="dark")


x=angles[:,2]
y=angles[:,0]
sns.scatterplot(x=x,y=y, s=5, color=".15")
# sns.histplot(x=x, y=y, bins=50, pthresh=-1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)

plt.gca().set_aspect('equal')
# %%
import random
f = random.choice(files)
structure = parser.get_structure('test_struct', f)
chains = list(structure.get_chains())
nv.show_biopython(chains[2])

# %%
new_structure = PDB.StructureBuilder()

# %%
def get_angles_lengths(chain):

    angles = np.empty((len(list(chain.get_residues())), 3))
    lengths = np.empty((len(list(chain.get_residues())), 4))
    angles[:] = np.nan
    lengths[:] = np.nan

    for i, r in enumerate(chain.get_residues()):
        try:
            if i != 0:
                angles[i-1, 0] = r.internal_coord.get_angle("omega")
                angles[i-1, 1] = r.internal_coord.get_angle("phi")

            lengths[i, 0] = r.internal_coord.get_length("N:CA")
            lengths[i, 1] = r.internal_coord.get_length("CA:C")
            lengths[i, 3] = r.internal_coord.get_length("C:O")

            if i != len(list(structure.get_residues()))-1:
                angles[i, 2] = r.internal_coord.get_angle("psi")
                lengths[i, 2] = r.internal_coord.get_length("C:1N")
        except AttributeError:
            pass

    return angles, lengths

def process_pdb_file(file):

    parser = PDB.PDBParser()
    structure = parser.get_structure('struct', file)
    structure.atom_to_internal_coordinates()

    angles= []
    lengths = []

    for c in structure.get_chains():
        a, l = get_angles_lengths(c)
        angles.append(a)
        lengths.append(l)
    
    return angles, lengths


angles = 5*[[]]
lengths = 5*[[]]
for f in tqdm(files[:10]):
    a, l = process_pdb_file(f)
    # angles += a
    # lengths += l
    print(len(a), len(l))
    for i in range(len(a)):
        angles[i].append(a[i])
        lengths[i].append(l[i])

# combined_angles = [np.concatenate(a, axis=0) for a in angles]
# combined_lengths = [np.concatenate(l, axis=0) for l in lengths]
# %%
sns.histplot([a.shape[0] for a in angles])

# %%

x=combined_angles[:, 1]
y=combined_angles[:, 2]
sns.scatterplot(x=x,y=y, s=3, alpha=0.3, color=".15")
plt.gca().set_aspect('equal')

# %%
sns.histplot(combined_angles[:, 0] % 360)
plt.xlim((90,270))

# %%
sns.histplot(combined_lengths[:, 0], color='red')
sns.histplot(combined_lengths[:, 1], color='green')
sns.histplot(combined_lengths[:, 2], color='blue')
sns.histplot(combined_lengths[:, 3], color='yellow')
plt.xlim((np.nanmin(combined_lengths), np.nanmax(combined_lengths)))
# %%

for f in tqdm(files[:10]):
    parser = PDB.PDBParser()
    structure = parser.get_structure('struct', f)
    structure.atom_to_internal_coordinates()

    print(len(list(structure.get_chains())))
# %%
