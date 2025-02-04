# DIM
## Authors: Namindu De Silva, Alberto Perez

Dynamic Ising Model (DIM) is a simple Dynamic Graphical Model (DGM) that calculates thermodynamics during B-to-A DNA structural transitions and generates a comprehensive transition matrix for nucleotide pucker properties giving insights on kinetics.
The model was trained using Ascona B-DNA consortium (ABC) simulation data to learn Dynamic coupling and bias parameters explained by S. Olsson and F. Noé (https://doi.org/10.1073/pnas.1901692116).

### Dependencies
The `dim` was trained extensively using `sklearn` and `graphtime`.
- python >= 3.6.1
- numpy >= 1.3
- itertools
- wheel

### Usage
1. Create conda environment
```
conda create -n <my-env> python=3.10
conda activate <my-env>
```

2. Clone the repository:
```
git clone https://github.com/PDNALab/DIM.git 
```

3. Install
```
cd DIM
python setup.py sdist bdist_wheel
cd dist
pip install dim-0.1.0-py3-none-any.whl
```

4. Main functions:
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