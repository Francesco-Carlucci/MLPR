import numpy as np
import numpy
import string
import scipy.special
import itertools
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
from itertools import repeat
import scipy.special as scsp
from util import loadPulsar,LDA,ZNormalization,plot_hist,plot_scatter,heatmap

if __name__ == '__main__':
    

    LDA_flag=False
    
    #load of training and test set with labels
    (DTR, LTR), (DTE, LTE)=loadPulsar()
    
    
    #Z normalization of data
    DTR, meanDTR, standardDeviationDTR=ZNormalization(DTR)
    DTE, meanDTE, standardDeviationDTE=ZNormalization(DTE)
    
        
    #linear discriminant analisis
    if LDA_flag:
        
        W=LDA(DTR,LTR,1)
        
        DTR=numpy.dot(W.T,DTR)
        DTE=numpy.dot(W.T,DTE)
        

    plot_hist(DTR, LTR, LDA_flag)
    #plot_scatter(DTR, LTR, LDA_flag)
    heatmap(DTR,LTR)
