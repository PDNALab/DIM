from . import utils as utils
import numpy as np
# import copy
# from joblib import Parallel, delayed

class dim:
    
    def __init__(self, seq, dmrf):
        self.seq = seq
        self.dmrf = dmrf
        self.coupling = self.get_coupling()
        self.bias = self.get_bias()
        # self.transition_matrix = self.get_transition_matrix()
        # self.free_energy2 = self.get_free_energy2()
        
    def get_coupling(self, seq:str=None, dmrf:list=None):
        if seq==None and dmrf==None:
            seq = self.seq
            dmrf = self.dmrf    
        return utils.coupling(seq=seq, dmrf=dmrf)
    
    def get_bias(self, seq:str=None, dmrf:list=None):
        if seq==None and dmrf==None:
            seq = self.seq
            dmrf = self.dmrf
        return utils.bias(seq=seq, dmrf=dmrf)
    
    def get_transition_matrix(self):
        return utils.get_transition_matrix(coupling=self.coupling, bias=self.bias)
    
    def get_free_energy1(self):
        return utils.free_energy1(coupling=self.coupling, bias=self.bias)
    
    def get_free_energy2(self):
        return utils.free_energy2(coupling=self.coupling, bias=self.bias, cut=10)
    
    def get_free_energy3(self):
        return utils.free_energy3(coupling=self.coupling, bias=self.bias)