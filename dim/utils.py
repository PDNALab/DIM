import numpy as np
import itertools
from joblib import Parallel, delayed

def complementary(seq:str):
    '''
    Following method returns complementary of a given sequence:
    (given 5-3 starnd, outputs complementary strnd from 5-3)
    '''
    seq = seq.upper()
    arr = []
    for base in list(seq):
        if base=='A':
            new_ = base.replace('A','T')
        elif base=='T':
            new_ = base.replace('T','A')
        elif base=='G':
            new_ = base.replace('G','C')
        elif base=='C':
            new_ = base.replace('C','G')
        else:
            raise ValueError('The sequence contains invalid characters.')
        arr.append(new_)
    return ''.join(arr)[::-1]


def check_(seq:str)->bool:
    if len(seq)<4:
        raise ValueError('The input sequence should be atleast 4 NA long.')
    else:
        seq = seq.upper()
        for s in seq:
            if s not in ['A','T','G','C']:
                raise ValueError('The sequence contains invalid characters.')
            


def coupling(seq:str, dmrf):
    '''
    Generate the coupling matrix by combining individual tetramer level coupling matrices
    '''
    len_ = len(seq)*2
    coupling = np.zeros((len_,len_),dtype=float)
    count = np.zeros((len_, len_), dtype=int)  # Count array to track how many times each cell is updated
    seq = 'C'+seq+'G'

    for i in range(0,len(seq)-3, 1):
        s = seq[i:i+4]
        # print(s)
        if s in dmrf.keys():
            c = np.mean([i.get_subsystem_couplings() for i in dmrf[s]], axis=0)
        elif complementary(s) in dmrf.keys():
            c = np.fliplr(np.flipud(np.mean([i.get_subsystem_couplings() for i in dmrf[complementary(s)]], axis=0)))
            
        coupling[i:i+2,i:i+2] += c[:2,:2] # update coupling
        count[i:i+2, i:i+2] += 1 # update count
        
        coupling[-(2+i):len_-i, -(2+i):len_-i] += c[2:,2:]
        count[-(2+i):len_-i, -(2+i):len_-i] += 1
        
        coupling[i:i+2, len_-(2+i):len_-i] += c[:2,2:]
        count[i:i+2, len_-(2+i):len_-i] += 1
        
        coupling[len_-(2+i):len_-i, i:i+2] += c[2:,:2]
        count[len_-(2+i):len_-i, i:i+2] += 1

    return np.divide(coupling, count, out=np.zeros_like(coupling), where=(count > 0))
    

def bias(seq:str, dmrf):
    len_ = len(seq)*2
    bias = np.zeros((len_),dtype=float)
    count = np.zeros((len_), dtype=int)  # Count array to track how many times each cell is updated
    seq = 'C'+seq+'G'

    for i in range(0,len(seq)-3, 1):
        s = seq[i:i+4]
        # print(s)
        if s in dmrf.keys():
            b = np.mean([i.get_subsystem_biases() for i in dmrf[s]], axis=0)
        elif complementary(s) in dmrf.keys():
            b = np.flip(np.mean([i.get_subsystem_biases() for i in dmrf[complementary(s)]], axis=0))
            
        bias[i:i+2] += b[:2] # update bias
        count[i:i+2] += 1 # update count
        
        bias[-(2+i):len_-i] += b[2:]
        count[-(2+i):len_-i] += 1

    return np.divide(bias, count, out=np.zeros_like(bias), where=(count > 0))

    
def get_combinations(n_subsystems):
    return np.array(list(itertools.product([-1, 1], repeat=n_subsystems)))

def get_subunit_states(n_subsystems:int, cut:int=10):
    '''
    n_subsystems: the number os total subsystems to be devided into several sub-units
    cut = the number of subsystems per sub-unit
    '''

    all_one = -np.ones(n_subsystems).astype(int)
    
    arr = []

    for i in range(int(n_subsystems/cut)+1 if n_subsystems%cut!=0 else int(n_subsystems/cut)):
        head_const = i*cut
        if (n_subsystems-(i+1)*cut) > 0:
            tail_const = head_const+cut
            states = np.array([tuple(all_one[0:head_const]) + combo + tuple(all_one[tail_const:]) for combo in itertools.product([-1,1],repeat=cut)])
            arr.append(states)
        else:
            states = np.array([tuple(all_one[0:head_const]) + combo for combo in itertools.product([-1,1],repeat=n_subsystems-i*cut)])
            arr.append(states)
        
    return arr

# @jit(nopython=True)
def theta(subsys, bias_index, coup, bias):
    return np.dot(coup[bias_index], subsys) + bias[bias_index]

# @jit(nopython=True)
def sub_proba(subsys, _subsys, coupling, bias):
    thetas = np.dot(coupling, subsys) + bias
    return 1 / (1 + np.exp(-_subsys * thetas))


def compute_transition_row(row, states, coupling, bias):
    num_states = len(states)
    state_row = states[row]

    thetas = np.dot(coupling, state_row) + bias
    row_data = np.zeros(num_states, dtype=np.float64)

    for col in range(num_states):
        state_col = states[col]
        row_data[col] = np.prod(1 / (1 + np.exp(-state_col * thetas)))

    return row_data

def get_transition_matrix(coupling, bias, states=None):
    '''
    Returns transition matrix given coupling and bias
    '''
    if states is None:
        states = get_combinations(bias.shape[0])
    num_states = len(states)

    # Parallelize the row computations
    results = Parallel(n_jobs=-1)(delayed(compute_transition_row)(row, states, coupling, bias) for row in range(num_states))

    # Create the transition matrix from the results
    TMat = np.array(results)

    return TMat


def get_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_distribution = eigenvectors[:, np.argmax(np.isclose(eigenvalues, 1))].real
    stationary_distribution /= stationary_distribution.sum()
    return stationary_distribution



def free_energy1(coupling, bias):
    '''
    Not suitable for systems with large number of sub-systems
    '''
    probs = get_stationary_distribution(get_transition_matrix(coupling=coupling, bias=bias))[[0,-1]]
    dG_NA = 8.314*(0.3/4.184)*(np.log(probs[0])-np.log(probs[1]))
    return dG_NA

def free_energy2(coupling, bias, cut=10): # kcal/mol
    '''
    Ideal for systems with large number of sub-systems
    '''
    states=get_subunit_states(n_subsystems=len(bias), cut=cut)
    dG_NA = 0
    for i in states:
        tmat = get_transition_matrix(coupling=coupling, bias=bias, states=i)
        for a in range(tmat.shape[0]):
            tmat[a,:] = tmat[a,:]/np.sum(tmat[a,:])
            
        probs = get_stationary_distribution(tmat)[[0,-1]] # get only stationary distrybutions of all -1 and all +1
        dG_NA += 8.314*(0.3/4.184)*(np.log(probs[0])-np.log(probs[1]))
        
    return dG_NA

def free_energy3(coupling, bias):
    '''
    returns free energy per each subsystem as a numpy array
    '''
    states = get_subunit_states(n_subsystems=len(bias), cut=1)
    arr = []
    for i in states:
        tmat = get_transition_matrix(coupling=coupling, bias=bias, states=i)
        for a in range(tmat.shape[0]):
            tmat[a,:] = tmat[a,:]/np.sum(tmat[a,:])
            
        probs = get_stationary_distribution(tmat)[[0,-1]] # get only stationary distrybutions of all -1 and all +1
        dG_NA = 8.314*(0.3/4.184)*(np.log(probs[0])-np.log(probs[1]))
        
        arr.append(dG_NA)
        
    return np.array(arr)