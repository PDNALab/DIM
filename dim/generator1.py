# from utils import complementary, check_, add_to_result
from . import utils
import numpy as np
import copy
from joblib import Parallel, delayed


class dim:
    
    def __init__(self, seq, dmrf):
        self.seq = seq
        self.dmrf = dmrf
        self.coupling = self.generate_coupling()
        self.bias = self.generate_bias()
    
    
    
    def generate_coupling(self, seq:str=None, dmrfs_tet:list=None):
        if seq==None and dmrfs_tet==None:
            seq = self.seq
            dmrfs_tet = self.dmrf
        utils.check_(seq)
        seq = 'GC'+seq+'GC'
        len_ = (len(seq)-4)*2
        result = np.zeros((len_, len_), dtype=float)
        counts = np.zeros((len_, len_), dtype=int)
        for s in range(len(seq)-3):
            if s == 0 or s == len(seq)-4:
                continue
            else:
                key_ = seq[s:s+4]
                if key_ in dmrfs_tet.keys():
                    arr = np.mean([i.get_subsystem_couplings() for i in dmrfs_tet[key_]], axis=0)
                elif utils.complementary(key_) in dmrfs_tet.keys():
                    arr = np.fliplr(np.flipud(np.mean([i.get_subsystem_couplings() for i in dmrfs_tet[utils.complementary(key_)]], axis=0)))
                i = s-1
                utils.add_to_result(result[i:i+2, i:i+2], counts[i:i+2, i:i+2], arr[0:2, 0:2])
                utils.add_to_result(result[i:i+2, len_-(2+i):len_-i], counts[i:i+2, len_-(2+i):len_-i], arr[0:2,2:4])
                utils.add_to_result(result[-(2+i):len_-i, -(2+i):len_-i], counts[-(2+i):len_-i, -(2+i):len_-i], arr[2:4, 2:4])
                utils.add_to_result(result[len_-(2+i):len_-i, i:i+2], counts[len_-(2+i):len_-i, i:i+2], arr[2:4,0:2])
        counts[counts == 0] = 1
        result = np.divide(result, counts)
        return result
    

    def generate_bias(self, seq:str=None, dmrfs_tet:list=None):
        if seq==None and dmrfs_tet==None:
            seq = self.seq
            dmrfs_tet = self.dmrf
        utils.check_(seq)
        seq = 'GC'+seq+'GC'
        len_ = (len(seq)-4)*2
        result = np.zeros((len_), dtype=float)
        counts = np.zeros((len_), dtype=int)
        for s in range(len(seq)-3):
            if s == 0 or s == len(seq)-4:
                continue
            else:
                key_ = seq[s:s+4]
                if key_ in dmrfs_tet.keys():
                    arr = np.mean([i.get_subsystem_biases() for i in dmrfs_tet[key_]], axis=0)
                elif utils.complementary(key_) in dmrfs_tet.keys():
                    arr = np.flip(np.mean([i.get_subsystem_biases() for i in dmrfs_tet[utils.complementary(key_)]], axis=0))
                i = s-1
                utils.add_to_result(result[i:i+2], counts[i:i+2], arr[0:2])
                utils.add_to_result(result[-(2+i):len_-i], counts[-(2+i):len_-i], arr[2:4])
        counts[counts == 0] = 1
        result = np.divide(result, counts)
        return result
    
    def get_transition_matrix(self):
        '''
        Returns transition matrix given coupling and bias
        '''
        return utils.get_transition_matrix(coupling=self.coupling, bias=self.bias)
    
    def get_free_energy(self, seq:str=None, dmrfs_tet:list=None):
        'Returns free energy of each NA going from south puckering to north puckering in a form of a numpy array'
        if seq==None and dmrfs_tet==None:
            seq = self.seq
            dmrfs_tet = self.dmrf
        len_ = len(seq)*2
        seq = 'G'+seq+'G'
        
        fe = np.zeros((len_), dtype=float)
        for n in range(0,len(seq)-3, 2):
            s = seq[n:n+4]
            if s in dmrfs_tet.keys():
                c = np.mean([i.get_subsystem_couplings() for i in dmrfs_tet[s]], axis=0)
                b = np.mean([i.get_subsystem_biases() for i in dmrfs_tet[s]], axis=0)
            elif utils.complementary(s) in dmrfs_tet.keys():
                c = np.fliplr(np.flipud(np.mean([i.get_subsystem_couplings() for i in dmrfs_tet[utils.complementary(s)]], axis=0)))
                b = np.flip(np.mean([i.get_subsystem_biases() for i in dmrfs_tet[utils.complementary(s)]], axis=0))
                
            dict_ = {}
            tm = utils.get_transition_matrix(coupling=c, bias=b)
            st_dis = utils.get_stationary_distribution(tm)
            for i in range(4):
                s_prob = utils.prob_south(nNA=i, stationary_distribution=st_dis, len_DNA=4)
                dG_NA = 8.314*(0.3/4.184)*(np.log(s_prob)-np.log(1-s_prob))
                dict_[i] = dG_NA
            
            val = list(dict_.values())
            fe[n] = val[0]
            fe[n+1] = val[1]
            fe[len_-n-1] = val[-1]
            fe[len_-n-2] = val[-2]
            
        return fe
    
    # def free_energy(self, div_n:int=8): # kcal/mol
    #     coupling = self.coupling
    #     bias = self.bias
    #     seq = self.seq #+ utils.complementary(self.seq)
    #     '''
    #     This function devide the molecule into sub sections and calculate the transition matrix for each part.
    #     Then calculates the free energy of S->N for each NA of the given seq.
        
    #     seq: string
    #         Complete NA sequence
    #     coupling: numpy array (n,n)
    #     bias: numpy array (n,)
    #     div_n: sequence length after slicing (time-efficient when kept below 12)
    #     '''
        
    #     dict_ = {}
        
    #     if div_n > len(seq):
    #         raise ValueError('div_n should be less than or equal to sequence length')
        
    #     if len(seq)<5:
    #         coupling = self.coupling
    #         bias = self.bias
            
    #         tm = utils.get_transition_matrix(coupling=coupling, bias=bias)
    #         st_dis = utils.get_stationary_distribution(tm)
    #         for i in range(len(seq)*2):
    #             s_prob = utils.prob_south(nNA=i, stationary_distribution=st_dis, len_DNA=len(seq)*2)
    #             dG_NA = 8.314*(0.3/4.184)*(np.log(s_prob)-np.log(1-s_prob))
    #             dict_[i] = dG_NA
                
    #         return dict_
        
    #     if len(seq) > 4:
    #         # we have to rearrange the coupling matrix and bias
        
        
        
        # rows_coup = coupling.shape[0]
        # parts_ = int(rows_coup/div_n)
        # n = 0
        # dict_ = {}
        # for i in range(1, parts_+2):
        #     if i*div_n<rows_coup:
        #         s = seq[(i-1)*div_n:i*div_n]
        #         # x = coupling[(i-1)*div_n:i*div_n,(i-1)*div_n:i*div_n]
        #         x = 
        #         xb = bias[(i-1)*div_n:i*div_n]
        #         tm = utils.get_transition_matrix(x,xb)
        #         st_dis = utils.get_stationary_distribution(tm)
                    
        #     else:
        #         s = seq[(i-1)*div_n:]
        #         x = coupling[(i-1)*div_n:,(i-1)*div_n:]
        #         xb = bias[(i-1)*div_n:]
        #         tm = utils.get_transition_matrix(x,xb)
        #         st_dis = utils.get_stationary_distribution(tm)
                    
        #     for i in range(len(s)):
        #         s_prob = utils.prob_south(nNA=i, stationary_distribution=st_dis, len_DNA=len(s))
        #         dG_NA = 8.314*(0.3/4.184)*(np.log(s_prob)-np.log(1-s_prob))
        #         dict_[n] = dG_NA
        #         n+=1

        # return dict_