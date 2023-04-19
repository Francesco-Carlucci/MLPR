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
import seaborn


PRIOR_PROBABILITY_TRUE_WHOLE_SET=1639/17898
PRIOR_PROBABILITY_TRUE_TRAINING_SET=821/8929

def vrow(v):
    return v.reshape((1,v.size))
def mcol(v):
    return v.reshape((v.size,1))

def loadPulsar():
    
    DTR = []
    LTR = []
    DTE = []
    LTE = []

    with open("Train.txt") as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = float(line.split(',')[-1].strip())
                DTR.append(attrs)
                LTR.append(label)
            except:
                pass
            
            
    with open("Test.txt") as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = float(line.split(',')[-1].strip())
                DTE.append(attrs)
                LTE.append(label)
            except:
                pass
            
    
    
    return (numpy.hstack(DTR), numpy.array(LTR, dtype=numpy.int32)), (numpy.hstack(DTE), numpy.array(LTE, dtype=numpy.int32))


def Ksplit(D, L, seed=0, K=3):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1]/K)
    # Generate a random seed
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    for i in range(K):
        folds.append(D[:,idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
        labels.append(L[idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
    return folds, labels


def ZNormalization(D, mean=None, standardDeviation=None):
    if (mean is None and standardDeviation is None):
        mean = D.mean(axis=1)
        standardDeviation = D.std(axis=1)
    ZD = (D-mcol(mean))/mcol(standardDeviation)
    return ZD, mean, standardDeviation

def plot_hist(D, L, PCA_flag):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'Mean of the integrated profile',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve',
        5: 'Standard deviation of the DM-SNR curve',
        6: 'Excess kurtosis of the DM-SNR curve',
        7: 'Skewness of the DM-SNR curve'        }

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel(('' if PCA_flag else hFea[dIdx]))
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label='Non-Pulsar')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label ='Pulsar')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()
    
    
def plot_scatter(D, L, PCA_flag):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'Mean of the integrated profile',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve',
        5: 'Standard deviation of the DM-SNR curve',
        6: 'Excess kurtosis of the DM-SNR curve',
        7: 'Skewness of the DM-SNR curve'        }

    for dIdx1 in range(D.shape[0]):
        for dIdx2 in range(D.shape[0]):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(('' if PCA_flag else hFea[dIdx1]))
            plt.ylabel(('' if PCA_flag else hFea[dIdx2]))
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Non-pulsar', s=1)
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Pulsar',s=1)

        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()
        

def heatmap(D, L):
    plt.figure()
    seaborn.heatmap(np.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==0]), linewidth=0.2, cmap="Reds", square=True,cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==1]), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    plt.show()
    return

def PCA(D,m):
    mu = D.mean(1)

    
    DC = D - mu.reshape((D.shape[0], 1))


    
    C = numpy.dot(DC, DC.T)/float(DC.shape[1])

    
    s, U = numpy.linalg.eigh(C)
    
    P = U[:, ::-1][:, 0:m]

    DP = numpy.dot(P.T, D)
    return DP

def covarianceMatrix(D):
    mu = D.mean(1)

    
    DC = D - mu.reshape((D.shape[0], 1))

    
    C = numpy.dot(DC, DC.T)/float(DC.shape[1])
    
    return C

def covarianceMatrixNaiveBayes(D):

    C=covarianceMatrix(D)
    
    C=C*numpy.identity(C.shape[1])
    
    return C


def LDA(D,L,m):
    DTR0 = D[:, L == 0]
    DTR1 = D[:, L == 1]
    CTR0 = covarianceMatrix(DTR0)
    CTR1 = covarianceMatrix(DTR1)
    SW = ((L.shape[0] - numpy.sum(L))*CTR0 +
          numpy.sum(L)*CTR1)/(L.shape[0])
    mu = D.mean(1)
    mu0 = DTR0.mean(1)
    mu1 = DTR1.mean(1)
    SB = ((L.shape[0] - numpy.sum(L))*numpy.dot(mcol((mu0-mu)), mcol((mu0-mu)).T) +
          numpy.sum(L)*numpy.dot(mcol((mu1-mu)), mcol((mu1-mu)).T))/(L.shape[0])
    # U, s, _ = numpy.linalg.svd(SW)
    # print(s)
    # P1 = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)
    # SBT = numpy.dot(numpy.dot(P1, SB), P1.T)
    # s, P2 = numpy.linalg.eigh(SBT)
    # print(s)
    # P2 = P2[:, ::-1][:, 0:m]
    # W = numpy.dot(P1.T, P2)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    
    return W

def logpdf_GAU_ND_single(X, mu, C):
    
    M=mu.shape[0]
    
    logpdf=-(M/2.0)*math.log(2.0*math.pi)  -(1/2.0)*numpy.linalg.slogdet(C)[1] -(1/2.0)*(numpy.dot(numpy.dot((X-mu).T,numpy.linalg.inv(C)),(X-mu)))
    
    logpdf=logpdf[0][0]
    
    return logpdf


def logpdf_GAU_ND(X,mu,C):
    
    logpdf=[]
    
    
    for i in range(X.shape[-1]):
        
        xn=X[...,i]

        xn=mcol(xn)
        mu=mcol(mu)

        logpdf.append(logpdf_GAU_ND_single(xn,mu,C))
        
    return logpdf

def calculatePrecision(LP,LT):
    
    total=LT.shape[0]
    correct=0
    
    if(total!=LP.shape[0]):
        print(total)
        print(LP.shape[0])
        raise Exception("Error not matching arrays!")
        
    for i in range(total):
        if LP[i]==LT[i]:
            correct+=1
    #correct=np.sum(LP==LT)
    
    return correct,total,(correct/total)
    


def GC(DTR, LTR, DTE, priorProbTrue=PRIOR_PROBABILITY_TRUE_TRAINING_SET):

    DTR0=DTR[:,LTR==0]
    DTR1=DTR[:,LTR==1]

    CTR0=covarianceMatrix(DTR0)
    CTR1=covarianceMatrix(DTR1)

    muTR0 = DTR0.mean(1)
    muTR1 = DTR1.mean(1)
    
    
    logpdf0=logpdf_GAU_ND(DTE,muTR0,CTR0)
    logpdf1=logpdf_GAU_ND(DTE,muTR1,CTR1)
    
    
    logS=[]
    
    logS=numpy.vstack([logpdf0,logpdf1])
    
    PriorP=numpy.vstack([math.log(1-priorProbTrue),math.log(priorProbTrue)])

    logSJoint=logS + PriorP
    
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))

    logSPost = logSJoint - logSMarginal
    #SPost = numpy.exp(logSPost)
    #SPost=numpy.argmax(SPost, 0)
    
    return logSPost


def naiveBayesGC(DTR, LTR, DTE, priorProbTrue=PRIOR_PROBABILITY_TRUE_TRAINING_SET):

    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    muTR0 = DTR0.mean(1)
    muTR1 = DTR1.mean(1)

    CTRNB0 = covarianceMatrixNaiveBayes(DTR0)
    CTRNB1 = covarianceMatrixNaiveBayes(DTR1)
    
    logpdfNB0=logpdf_GAU_ND(DTE,muTR0,CTRNB0)
    logpdfNB1=logpdf_GAU_ND(DTE,muTR1,CTRNB1)
    
    logNBS=[]
    
    logNBS=numpy.vstack([logpdfNB0,logpdfNB1])
    
    PriorP=numpy.vstack([math.log(1-priorProbTrue),math.log(priorProbTrue)])

    logNBSJoint=logNBS + PriorP
    
    logNBSMarginal = vrow(scipy.special.logsumexp(logNBSJoint, axis=0))

    logNBSPost = logNBSJoint - logNBSMarginal
    #SPost = numpy.exp(logNBSPost)
    #SPost=numpy.argmax(SPost, 0)

    return logNBSPost

def tiedCovarianceGC(DTR, LTR, DTE, priorProbTrue=PRIOR_PROBABILITY_TRUE_TRAINING_SET):

    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    muTR0 = DTR0.mean(1)
    muTR1 = DTR1.mean(1)


    CTR0=covarianceMatrix(DTR0)
    CTR1=covarianceMatrix(DTR1)

    
    tot=LTR.shape[0]
    num1=numpy.sum(LTR)
    num0=tot-num1
    
    CTC= (num0*CTR0 + num1*CTR1)/tot
    
    logpdf0_TC=logpdf_GAU_ND(DTE,muTR0,CTC)
    logpdf1_TC=logpdf_GAU_ND(DTE,muTR1,CTC)

    
    logS_TC=[]
    
    logS_TC=numpy.vstack([logpdf0_TC,logpdf1_TC])
    
    PriorP=numpy.vstack([math.log(1-priorProbTrue),math.log(priorProbTrue)])

    logSJoint_TC=logS_TC + PriorP
    
    logSMarginal_TC = vrow(scipy.special.logsumexp(logSJoint_TC, axis=0))

    logSPost_TC = logSJoint_TC - logSMarginal_TC
    #SPost_TC = numpy.exp(logSPost_TC)
    #SPost_TC=numpy.argmax(SPost_TC, 0)

    return logSPost_TC
    
    
def tiedCovarianceNaiveBayesGC(DTR, LTR, DTE, priorProbTrue=PRIOR_PROBABILITY_TRUE_TRAINING_SET):
    
    DTR0=DTR[:,LTR==0]
    DTR1=DTR[:,LTR==1]

    muTR0 = DTR0.mean(1)
    muTR1 = DTR1.mean(1)
    
    CTR0=covarianceMatrix(DTR0)
    CTR1=covarianceMatrix(DTR1)
    
    tot=LTR.shape[0]
    num1=numpy.sum(LTR)
    num0=tot-num1
    
    CTC= (num0*CTR0 + num1*CTR1)/tot
    
    CTC=CTC*numpy.identity(CTC.shape[1])
    
    
    logpdf0_TC=logpdf_GAU_ND(DTE,muTR0,CTC)
    logpdf1_TC=logpdf_GAU_ND(DTE,muTR1,CTC)

    logS_TC=[]
    
    logS_TC=numpy.vstack([logpdf0_TC,logpdf1_TC])
    
    PriorP=numpy.vstack([math.log(1-priorProbTrue),math.log(priorProbTrue)])

    logSJoint_TC=logS_TC + PriorP
    
    logSMarginal_TC = vrow(scipy.special.logsumexp(logSJoint_TC, axis=0))

    logSPost_TC = logSJoint_TC - logSMarginal_TC
    #SPost_TC = numpy.exp(logSPost_TC)
    #SPost_TC=numpy.argmax(SPost_TC, 0)

    return logSPost_TC

def calculateConfusionMatrix(PL,LTE):
    
    Conf=numpy.zeros((2,2))

    for i in range(LTE.shape[0]):
        x=PL[i]
        y=LTE[i]
        Conf[x][y]+=1
                
    return Conf

def calculateConfusionMatrixSVM(PL,LTE):
    
    Conf=numpy.zeros((2,2))
    
    for i in range(LTE.shape[0]):
        x=PL[i]
        y=LTE[i]
        if x==-1:
            x=0
        if y==-1:
            y=0
        Conf[x][y]+=1
    return Conf  
    
    

def logreg_obj(v, DTR, LTR, l):
    
    w, b = v[0:-1], v[-1]
    
    w_norm=numpy.linalg.norm(w)
    
    
    J=0

    
    for i in range(DTR.shape[1]):
        
        if(LTR[i]==1):
            z_i=1
        else:
            z_i=-1
            
        J=J + numpy.logaddexp(0, -z_i*(numpy.dot(w.T,DTR[:,i])+ b))


    J=(J/DTR.shape[1])+(l/2)*w_norm**2
    
    
    return J

def binaryLogisticRegression(DTR,LTR,DTE,lamb):
    
    x0 = numpy.zeros(DTR.shape[0] + 1)
    
    x, f, d=scipy.optimize.fmin_l_bfgs_b(logreg_obj,x0,args=(DTR, LTR, lamb),iprint = 1,approx_grad = True)

    w, b = x[0:-1], x[-1]

    S=[]

    for i in range(DTE.shape[1]):
    
        S.append(numpy.dot(w.T,DTE[:,i]) + b)
    
    return S

def computeBLRThreshold(Cfn,Cfp,pi1=PRIOR_PROBABILITY_TRUE_TRAINING_SET):
    
    t=-math.log(pi1*Cfn/((1-pi1)*Cfp))
    
    return t
    
def computeDCF(C,Cfn,Cfp,pi1=PRIOR_PROBABILITY_TRUE_TRAINING_SET):
    
    FNR=C[0][1]/(C[0][1]+C[1][1])
    FPR=C[1][0]/(C[1][0]+C[0][0])
    
    DCF=pi1*Cfn*FNR + (1-pi1)*Cfp*FPR
    
    return DCF

def computeBDummy(Cfn,Cfp,pi1=PRIOR_PROBABILITY_TRUE_TRAINING_SET):
    
    c1=pi1*Cfn
    c2=(1-pi1)*Cfp
    
    if(c1<c2):
        return c1
    else:
        return c2
   


# def supportVectorMachine(DTR,LTR,DTE,LTE,K,C):
    
#     N = DTR.shape[1]
#     F = DTR.shape[0]
    
#     LTRz = np.zeros(N)
#     for i in range(N):
#         LTRz[i] = 1 if LTR[i]==1 else -1
        
#     # Compute the expaded feature space D_ 
#     D_ = np.vstack((DTR, K*np.ones(N)))
    
#     # Compute matrix G_ of dot products of all samples of D_
#     G_ = np.dot(D_.T, D_)
    
#     # Compute matrix H_
#     LTRz_matrix = np.dot(LTRz.reshape(-1,1), LTRz.reshape(1,-1))
#     H_ = G_ * LTRz_matrix
    
#     def gradLDc(alpha):
#         n = len(alpha)
#         return (np.dot(H_, alpha) - 1).reshape(n)
    
#     # Define a function that represents J_D(alpha) we want to minimize
#     def LDc_obj(alpha): # alpha has shape (n,)
#         n = len(alpha)
#         minusJDc = 0.5 * np.dot(np.dot(alpha.T, H_), alpha) - np.dot(alpha.T, np.ones(n)) # 1x1
#         return minusJDc, gradLDc(alpha)
    
#     # Minimize LD_(alpha)
#     bounds = [(0,C)] * N
#     m, primal, _ = scipy.optimize.fmin_l_bfgs_b(func=LDc_obj,
#                                            bounds=bounds,
#                                            x0=np.zeros(N), factr=1.0)
#     # m is the final alpha
#     wc_star = np.sum(m * LTRz * D_, axis=1)
    
#     # extract w and b
#     w_star, b_star = wc_star[:-1], wc_star[-1]
    
#     # compute the scores
#     S = np.dot(w_star.T, DTE) + b_star*K # the *K is not present in slides!??
#     # or: S=np.dot(wc_star.T, np.vstack((DTE, K*np.ones(DTE.shape[1]))))
    
#     def primal_obj(wc_star):
#         return 0.5 * np.linalg.norm(wc_star)**2 + C * np.sum(np.maximum(0,1-LTRz * np.dot(wc_star.T, D_)))
#     primal_loss = primal_obj(wc_star)
#     dual_loss = LDc_obj(m)[0]
#     duality_gap=primal_loss + dual_loss
    
#     predict_labels = np.where(S > 0, 1, 0)
    
#     correctlyClassified=sum(predict_labels == LTE)
#     tot=len(predict_labels)
    
#     acc = correctlyClassified / tot
#     conf=calculateConfusionMatrix(predict_labels,LTE)
#     print('The Linear Support Vector Machine with parameters: C=%.1f, K=%d classifies correctly %d labels of %d. Primal loss: %f, Dual loss: %f, Duality gap: %.9f, Error rate: %.1f%%'%(C,K,correctlyClassified,tot,primal_loss,dual_loss,duality_gap,(1-acc)*100))
#     print("Confusion matrix:")
#     print(conf)
    
#     return
    

def LD_objectiveFunctionOfModifiedDualFormulation(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2)*np.dot(np.dot(alpha.T, H), alpha)-np.dot(alpha.T, np.ones(H.shape[1])), grad)


def primalObjective(w, D, C, LTR, f):
    normTerm = (1/2)*(np.linalg.norm(w)**2)
    m = np.zeros(LTR.size)
    for i in range(LTR.size):
        vett = [0, 1-LTR[i]*(np.dot(w.T, D[:, i]))]
        m[i] = vett[np.argmax(vett)]
    pl = normTerm + C*np.sum(m)
    dl = -f
    dg = pl-dl
    return pl, dl, dg


def primalLossDualLossDualityGapErrorRate(DTR, C, Hij, LTR, LTE, DTE, D, K):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b,  iprint=1, factr=1.0)
    # Now we can recover the primal solution
    w = np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)
    # Compute the scores as in the previous lab
    S = np.dot(w.T, DTE)
    # Compute predicted labels. 1* is useful to convert True/False to 1/0
    LP = 1*(S > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # Compute primal loss, dual loss, duality gap
    pl, dl, dg = primalObjective(w, D, C, LTR, f)
    conf=calculateConfusionMatrixSVM(LP,LTE)
    print("Printing to file...")
    with open('SVM_DUMP.txt', 'a') as f:
        print("K=%d, C=%f, Primal loss=%e, Dual loss=%e, Duality gap=%e, Error rate=%.1f %%" % (
            K, C, pl, dl, dg, errorRate), file=f)
        print("Confusion matrix:", file=f)
        print(conf, file=f)
    return


def modifiedDualFormulation(DTR, LTR, DTE, LTE, K):
    # Compute the D matrix for the extended training set with K=1
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])
    row = np.zeros(DTE.shape[1])+K
    DTE = np.vstack([DTE, row])
    # Compute the H matrix exploiting broadcasting
    Gij = np.dot(D.T, D)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    # We use three different values for hyperparameter C
    # 1) C=0.1
    primalLossDualLossDualityGapErrorRate(DTR, 0.1, Hij, LTR, LTE, DTE, D, K)
    # 2) C=1
    primalLossDualLossDualityGapErrorRate(DTR, 1, Hij, LTR, LTE, DTE, D, K)
    # 3) C=10
    primalLossDualLossDualityGapErrorRate(DTR, 10, Hij, LTR, LTE, DTE, D, K)
    return


def dualLossErrorRatePoly(DTR, C, Hij, LTR, LTE, DTE, K, d, c):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    # Compute the scores
    S = np.sum(
        np.dot((x*LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DTE)+c)**d+ K), axis=0)
    # Compute predicted labels. 1* is useful to convert True/False to 1/0
    LP = 1*(S > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # Compute dual loss
    dl = -f
    conf=calculateConfusionMatrixSVM(LP,LTE)
    print("Printing to file...")
    with open('SVM_DUMP.txt', 'a') as f:
        print("K=%d, C=%f, Kernel Poly (d=%d, c=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, d, c, dl, errorRate), file=f)
        print("Confusion matrix:", file=f)
        print(conf, file=f)
    return

def dualLossErrorRateRBF(DTR, C, Hij, LTR, LTE, DTE, K, gamma):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    kernelFunction = np.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTE[:, j], gamma, K)
    S=np.sum(np.dot((x*LTR).reshape(1, DTR.shape[1]), kernelFunction), axis=0)
    # Compute the scores
    # S = np.sum(
    #     np.dot((x*LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DTE)+c)**d+ K), axis=0)
    # Compute predicted labels. 1* is useful to convert True/False to 1/0
    LP = 1*(S > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # Compute dual loss
    dl = -f
    conf=calculateConfusionMatrixSVM(LP,LTE)
    print("Printing to file...")
    with open('SVM_DUMP.txt', 'a') as f:
        print("K=%d, C=%f, RBF (gamma=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, gamma, dl, errorRate), file=f)
        print("Confusion matrix:", file=f)
        print(conf, file=f)
    return


def kernelPoly(DTR, LTR, DTE, LTE, K, C, d, c):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = (np.dot(DTR.T, DTR)+c)**d+ K**2
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    dualLossErrorRatePoly(DTR, C, Hij, LTR, LTE, DTE, K, d, c)
    return


def RBF(x1, x2, gamma, K):
    return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))+K**2

def kernelRBF(DTR, LTR, DTE, LTE, K, C, gamma):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTR[:, j], gamma, K)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    dualLossErrorRateRBF(DTR, C, Hij, LTR, LTE, DTE, K, gamma)
    return



def logpdf_GMM(X, gmm):
    # This function will compute the log-density of a GMM for a set of samples
    # contained in matrix X of shape (D, N), where D is the size of a sample
    # and N is the number of samples in X.
    # We define a matrix S with shape (M, N). Each row will contain the sub-class
    # conditional densities given component Gi for all samples xi
    S = np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        # Compute log density
        S[i, :] = logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
        # Add log of the prior of the corresponding component
        S[i, :] += np.log(gmm[i][0])
    # Compute the log-marginal log fxi(xi). The result will be an array of shape
    # (N,) whose component i will contain the log-density for sample xi
    logdens = scsp.logsumexp(S, axis=0)
    return (logdens, S)


def Estep(logdens, S):
    # E-step: compute the POSTERIOR PROBABILITY (=responsibilities) for each component of the GMM
    # for each sample, using the previous estimate of the model parameters.
    return np.exp(S-logdens.reshape(1, logdens.size))


def constrainSigma(sigma):
    psi = 0.01
    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, mcol(s)*U.T)
    return sigma


def DiagConstrainSigma(sigma):
    sigma = sigma * np.eye(sigma.shape[0])
    psi = 0.01
    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, mcol(s)*U.T)
    return sigma


def Mstep(X, S, posterior):
    psi = 0.01
    # M-step: update the model parameters.
    Zg = np.sum(posterior, axis=1)  # 3
    # print(Zg)
    # Fg = np.array([np.sum(posterior[0, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[1, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[2, :].reshape(1, posterior.shape[1])*X, axis=1)])
    # print(Fg)
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    # print(Fg)
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * np.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    # print(Sg)
    mu = Fg / Zg
    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    # print(prodmu)
    # print(np.dot(mu, mu.T).reshape((1, mu.shape[0], mu.shape[0]))) NO, it is wrong
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    for g in range(S.shape[0]):
        U, s, Vh = np.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = np.dot(U, mcol(s)*U.T)
    # print(cov)
    w = Zg/np.sum(Zg)
    # print(w)
    return (w, mu, cov)


def DiagMstep(X, S, posterior):
    psi = 0.01
    # M-step: update the model parameters.
    Zg = np.sum(posterior, axis=1)  # 3
    # print(Zg)
    # Fg = np.array([np.sum(posterior[0, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[1, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[2, :].reshape(1, posterior.shape[1])*X, axis=1)])
    # print(Fg)
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    # print(Fg)
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * np.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    # print(Sg)
    mu = Fg / Zg
    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    # print(prodmu)
    # print(np.dot(mu, mu.T).reshape((1, mu.shape[0], mu.shape[0]))) NO, it is wrong
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    for g in range(S.shape[0]):
        cov[g] = cov[g] * np.eye(cov[g].shape[0])
        U, s, Vh = np.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = np.dot(U, mcol(s)*U.T)
    # print(cov)
    w = Zg/np.sum(Zg)
    # print(w)
    return (w, mu, cov)


def TiedMstep(X, S, posterior):
    psi = 0.01
    # M-step: update the model parameters.
    Zg = np.sum(posterior, axis=1)  # 3
    # print(Zg)
    # Fg = np.array([np.sum(posterior[0, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[1, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[2, :].reshape(1, posterior.shape[1])*X, axis=1)])
    # print(Fg)
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    # print(Fg)
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * np.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    # print(Sg)
    mu = Fg / Zg
    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    # print(prodmu)
    # print(np.dot(mu, mu.T).reshape((1, mu.shape[0], mu.shape[0]))) NO, it is wrong
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    tsum = np.zeros((cov.shape[1], cov.shape[2]))
    for g in range(S.shape[0]):
        tsum += Zg[g]*cov[g]
    for g in range(S.shape[0]):
        cov[g] = 1/X.shape[1] * tsum
        U, s, Vh = np.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = np.dot(U, mcol(s)*U.T)
    # print(cov)
    w = Zg/np.sum(Zg)
    # print(w)
    return (w, mu, cov)


def EMalgorithm(X, gmm, solutionFile):
    # The algorithm consists of two steps, E-step and M-step
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = np.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        posterior = Estep(logdens, S)
        # ------ M-step ----------
        (w, mu, cov) = Mstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikelihood2 = np.sum(logdens)/X.shape[1]
        if (loglikelihood2-loglikelihood1 < 10**(-6)):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    # print(count)
    #print("AVG log likelihood:", loglikelihood2)   
    return gmm


def DiagEMalgorithm(X, gmm, solutionFile):
    # The algorithm consists of two steps, E-step and M-step
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = np.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        posterior = Estep(logdens, S)
        # ------ M-step ----------
        (w, mu, cov) = DiagMstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikelihood2 = np.sum(logdens)/X.shape[1]
        if (loglikelihood2-loglikelihood1 < 10**(-6)):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    # print(count)
    #print("AVG log likelihood:", loglikelihood2)
    return gmm


def TiedEMalgorithm(X, gmm, solutionFile):
    # The algorithm consists of two steps, E-step and M-step
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = np.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        posterior = Estep(logdens, S)
        # ------ M-step ----------
        (w, mu, cov) = TiedMstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikelihood2 = np.sum(logdens)/X.shape[1]
        if (loglikelihood2-loglikelihood1 < 10**(-6)):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    # print(count)
    #print("AVG log likelihood:", loglikelihood2)
    return gmm


def GAU_logpdf(x, mu, var):
    # Function that computes the log-density of the dataset and returns it as a
    # 1-dim array
    return (-0.5*np.log(2*np.pi))-0.5*np.log(var)-(((x-mu)**2)/(2*var))


def plotNormalDensityOverNormalizedHistogram(dataset, gmm):
    # Function used to plot the computed normal density over the normalized histogram
    plt.figure()
    plt.hist(dataset, bins=30, edgecolor='black', linewidth=0.5, density=True)
    # Define an array of equidistant 1000 elements between -10 and 5
    XPlot = np.linspace(-10, 5, 1000)
    # We should plot the density, not the log-density, so we need to use np.exp
    y = np.zeros(1000)
    for i in range(len(gmm)):
        y += gmm[i][0]*np.exp(GAU_logpdf(XPlot, gmm[i]
                              [1], gmm[i][2])).flatten()
    plt.plot(XPlot, y,
             color="red", linewidth=3)
    return


def split(GMM):
    alpha = 0.1
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = np.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    # print("Splitted GMM", splittedGMM)
    return splittedGMM


def LBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm(X, GMM, '')
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm(X, GMM, '')
    return GMM


def DiagLBGalgorithm(GMM, X, iterations):
    GMM = DiagEMalgorithm(X, GMM, '')
    for i in range(iterations):
        GMM = split(GMM)
        GMM = DiagEMalgorithm(X, GMM, '')
    return GMM


def TiedLBGalgorithm(GMM, X, iterations):
    GMM = TiedEMalgorithm(X, GMM, '')
    for i in range(iterations):
        GMM = split(GMM)
        GMM = TiedEMalgorithm(X, GMM, '')
    return GMM


def performClassification(DTR0, DTR1, DEV, algorithm, K, LEV, constrain):
    # Define a list that includes the three splitted training set
    D = [DTR0, DTR1]
    # Define a list to store marginal likelihoods for the three sets
    marginalLikelihoods = []
    # Iterate on the three sets
    for i in range(len(D)):
        wg = 1.0
        # Find mean and covariance matrices, reshape them as matrices because they
        # are scalar and in the following we need them as matrices
        mug = D[i].mean(axis=1).reshape((D[i].shape[0], 1))
        sigmag = constrain(np.cov(D[i]).reshape(
            (D[i].shape[0], D[i].shape[0])))
        # Define initial component
        initialGMM = [(wg, mug, sigmag)]
        finalGMM = algorithm(initialGMM, D[i], K)
        # Compute marginal likelihoods and append them to the list
        marginalLikelihoods.append(logpdf_GMM(DEV, finalGMM)[0])
    # Stack all the likelihoods in PD
    PD = np.vstack(
        (marginalLikelihoods[0], marginalLikelihoods[1]))
    # Compute the predicted labels
    predictedLabels = np.argmax(PD, axis=0)
    numberOfCorrectPredictions = np.array(predictedLabels == LEV).sum()
    accuracy = numberOfCorrectPredictions/LEV.size*100
    errorRate = 100-accuracy
    return errorRate

