# DIM
## Authors: Namindu De Silva, Alberto Perez

Dynamic Ising Model (DIM) is a simple Dynamic Graphical Model (DGM) that calculates thermodynamics during B-to-A DNA structural transitions and generates a comprehensive transition matrix for nucleotide pucker properties giving insights on kinetics.
The model was trained using Ascona B-DNA consortium (ABC) simulation data to learn Dynamic coupling and bias parameters explained by S. Olsson and F. Noé (https://doi.org/10.1073/pnas.1901692116).

### Dependencies
The `dim` library makes extensive use of `numpy` and `sklearn` and `graphtime`.
- python >= 3.6.1
- numpy >= 1.3
- scikit-learn >= 0.19.0
- scipy >= 1.1.0
- msmtools >= 1.2.1
- pyemma >= 2.5.2
- pickel >= 4.0
- graphtime

### Usage
1. Clone the repository:
```
git clone https://github.com/PDNALab/DIM.git 
```

2. Append path: Befor importing DIM module, please append the path as shown.
```
import sys
sys.path.append('<path to dim>')

from dim import utils
from dim import dimgen as dim 
```

3. Main functions:
- Make dim object for arbitary DNA sequence. [sequence is given 5'-3']
```
# Load DMRF object - pre learned from ABC data
with open('<path to dim>/dim/gen_data/dmrf_tetramer_20_4.dmrf', 'rb') as f:
    dmrf = pickle.load(f)

# Make dim object
DNA = dim.dim(seq='ATGCATGC', dmrf=dmrf)
```
- Free energy
```
# For smaller DNA sequences:
Free_energy1 = DNA.get_free_energy1()

# When DNA sequences have more sub-systems [faster method]:
Free_energy2 = get_free_energy2(cut=10)
```
- Transition matrix
```
T_mat = DNA.get_transition_matrix()
```