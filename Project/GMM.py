import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from util import loadPulsar,PCA,ZNormalization,Ksplit
import time

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

def logpdf_GAU_ND(x, mu, C):  # C is sigma,covariance matrix
    _, logdetC = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    D=x.shape[0]
    #Mat=0.5*np.dot(np.dot((x-mu).T,invC),(x-mu)).diagonal()
    Mat = 0.5 * np.multiply(np.dot((x - mu).T, invC).T, (x - mu)).sum(axis=0)
    res2=-D/2*np.log(2*np.pi)-0.5*logdetC-Mat

    return np.array(res2)

def logpdf_GMM(X,gmm):
    M=len(gmm)
    S=[]
    for g in range(M):
        S.append(logpdf_GAU_ND(X,gmm[g][1],gmm[g][2])+np.log(gmm[g][0]))
    S=np.vstack(S)  #matrix of the joint likelihood

    logdens=scipy.special.logsumexp(S,axis=0) #vector of the log-marginal densities
    return logdens   #marginal densities

def computeLogG(X,gmm):
    M = len(gmm)
    S = []
    for g in range(M):
        S.append(logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0]))
    S = np.vstack(S)  # matrix of the joint likelihood [SJoint]
    logdens = scipy.special.logsumexp(S, axis=0)  # [SMarginal] vector of the log-marginal densities

    G=S-logdens   #/:/ [SPost], vector of responsibilities

    return G

def splitGMM(GMM_n,alpha):
    GMM_n1 = []
    for g in range(len(GMM_n)):
        U, s, Vh = np.linalg.svd(GMM_n[g][2])  # GMM_1[g][2]
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        GMM_n1.append((GMM_n[g][0] / 2, GMM_n[g][1] + d, GMM_n[g][2]))
        GMM_n1.append((GMM_n[g][0] / 2, GMM_n[g][1] - d, GMM_n[g][2]))
    return GMM_n1

def constrainCovMat(C,M,psi):
    for g in range(M):
        U, s, _ = np.linalg.svd(C[g])
        s[s < psi] = psi
        C[g] = np.dot(U, vcol(s) * U.T)
    return C

def compute_mu_s(GMMdata):
    mu = np.array(vcol(GMMdata.mean(1)))
    centerData = GMMdata - mu
    S = np.dot(centerData, centerData.T) / centerData.shape[1]

    return mu, S

def computellr(DTE,GMMparams0,GMMparams1):
    ll0 = logpdf_GMM(DTE, GMMparams0)  # compute marginals of each class
    ll1 = logpdf_GMM(DTE, GMMparams1)

    llr=ll1-ll0

    return llr

def LBGAlgorithm(GMMdata,mu,S,EMStop,alpha,splitnum,diag,tied):
    GMM_n = EM_GMMparams(GMMdata, [(1.0, mu, S)], EMStop)
    GMMList=[]

    for i in range(splitnum):
        GMM_n = splitGMM(GMM_n, alpha)

        GMM_nEM = EM_GMMparams(GMMdata, GMM_n, EMStop,diag,tied)  #diag,tied flags
        GMMList.append(GMM_nEM)

    return GMMList

def EM_GMMparams(GMMdata,GMMparams,EMStop,diagFlag=0,tiedFlag=0):
    avgll = 0
    llDiff = 100
    psi=0.01

    M = len(GMMparams)
    N = GMMdata.shape[1]

    while llDiff>EMStop:
        G=np.exp(computeLogG(GMMdata,GMMparams))
        #statistics
        Zg=vcol(G.sum(axis=1))
        Fg=np.dot(G,GMMdata.T)
        Sg=[np.dot(G[i]*GMMdata,GMMdata.T) for i in range(M)]
        #new GMM parameters
        munext=Fg/Zg
        Cnext=[Sg[i]/Zg[i]-np.dot(vcol(munext[i]),vrow(munext[i])) for i in range(M)]
        wnext=Zg/Zg.sum(axis=0)

        if diagFlag:
            #diagonal covariance matrix
            for g in range(M):
                Cnext[g]=Cnext[g]*np.eye(Cnext[g].shape[0])

        if tiedFlag:
            #Tied covariance matrix
            tiedCnext=np.sum([(Zg[g]*Cnext[g])for g in range(M)],axis=0)/N

            for g in range(M):
                Cnext[g]=tiedCnext

        #Covariance matrix eigenvalue constraint
        Cnext=constrainCovMat(Cnext,M,psi)

        GMMparams=[(wnext[i],vcol(munext[i]),Cnext[i]) for i in range(M)]

        newAvgll=logpdf_GMM(GMMdata,GMMparams).sum()/N
        llDiff=np.abs(newAvgll-avgll)
        avgll=newAvgll

        print('log likelihood: ',avgll)

    return GMMparams
"""
def KfoldLBG(folds, labels,i):
    trainingSet = []
    labelsOfTrainingSet = []
    for j in range(k):
        if j != i:
            trainingSet.append(folds[j])
            labelsOfTrainingSet.append(labels[j])
    evaluationSet = folds[i]
    orderedLabels.append(labels[i])
    trainingSet = np.hstack(trainingSet)
    labelsOfTrainingSet = np.hstack(labelsOfTrainingSet)
    trainSet0 = trainingSet[:, labelsOfTrainingSet == 0]
    trainSet1 = trainingSet[:, labelsOfTrainingSet == 1]
    mu0, S0 = compute_mu_s(trainSet0)
    mu1, S1 = compute_mu_s(trainSet1)
    GMMList0 = LBGAlgorithm(trainSet0, mu0, constrainCovMat([S0], 1, psi)[0], EMStop, alpha, splitnum, diag, tied)
    GMMList1 = LBGAlgorithm(trainSet1, mu1, constrainCovMat([S1], 1, psi)[0], EMStop, alpha, splitnum, diag, tied)

    return computellr(evaluationSet, GMMList0[splitIdx], GMMList1[splitIdx]), i
"""
def GMMClf(DTR,LTR,M,EMStop=1e-6,alpha=0.1,psi=0.01, diag=0,tied=0 ):

    splitnum = int(np.log2(M))
    #GMMdata0=DTR[:,LTR==0]
    #GMMdata1=DTR[:,LTR==1]
    #mu0,S0=compute_mu_s(GMMdata0)
    #mu1, S1= compute_mu_s(GMMdata1)

    k=3
    folds, labels = Ksplit(DTR, LTR, seed=0, K=3)
    orderedLabels = []
    scores = [[] for i in range(splitnum)]
    # if PCA_flag:
    for i in range(k):
        # trainingSet=[folds[j] for j in range(k) if j!=i]
        trainingSet = []
        labelsOfTrainingSet = []
        for j in range(k):
            if j != i:
                trainingSet.append(folds[j])
                labelsOfTrainingSet.append(labels[j])
        evaluationSet = folds[i]
        orderedLabels.append(labels[i])
        trainingSet = np.hstack(trainingSet)
        labelsOfTrainingSet = np.hstack(labelsOfTrainingSet)
        trainSet0=trainingSet[:,labelsOfTrainingSet==0]
        trainSet1=trainingSet[:, labelsOfTrainingSet == 1]
        mu0, S0 = compute_mu_s(trainSet0)
        mu1, S1 = compute_mu_s(trainSet1)
        GMMList0 = LBGAlgorithm(trainSet0, mu0, constrainCovMat([S0],1,psi)[0], EMStop, alpha, splitnum, diag, tied)
        GMMList1 = LBGAlgorithm(trainSet1, mu1, constrainCovMat([S1],1,psi)[0], EMStop, alpha, splitnum, diag, tied)

        for splitIdx in range(splitnum):
            #crea vettore riga degli score di ogni split
            scores[splitIdx].append(computellr(evaluationSet, GMMList0[splitIdx], GMMList1[splitIdx]))
        

    orderedLabels = np.hstack(orderedLabels)

    dcf1=[]
    dcf5=[]
    dcf9=[]
    for splitIdx in range(splitnum):
        dcf1.append(computeMinDCF(0.1, 1, 1, np.hstack(scores[splitIdx]), orderedLabels))
        dcf5.append(computeMinDCF(0.5, 1, 1, np.hstack(scores[splitIdx]), orderedLabels))
        dcf9.append(computeMinDCF(0.9, 1, 1, np.hstack(scores[splitIdx]), orderedLabels))

    return dcf1,dcf5,dcf9

def computeMinDCF(p1,Cfn,Cfp,llr,labels):
    nClasses=max(labels)+1
    Bnorm = min(p1 * Cfn, (1 - p1) * Cfp)
    pred_t = [llr > t for t in sorted(llr)]
    M_t = [confMat(pred_t[i], labels, nClasses) for i in range(len(llr))]
    DCF_t = [computeBayesRisk(p1, Cfn, Cfp, M_t[i]) for i in range(len(llr))]
    normDCF_t = np.array(DCF_t) / Bnorm

    return min(normDCF_t)
def computeBayesRisk(p1,Cfn,Cfp,M):
    FPR=M[1,0]/(M[0,0]+M[1,0])
    FNR=M[0,1]/(M[0,1]+M[1,1])
    DCF=p1*Cfn*FNR+(1-p1)*Cfp*FPR

    return DCF
def confMat(predLabels,labels,nClasses):
    confMat = np.zeros((nClasses, nClasses))
    #preCor = [(predLabels[i], LTE[i]) for i in range(predLabels.shape[0])]
    for i in range(nClasses):
        predH = predLabels == i
        for j in range(nClasses):
            corrH = labels == j
            confMat[i, j] = np.dot(predH.astype(int), corrH.astype(int).T)
    return confMat

if __name__=="__main__":
    PCA_flag = False
    PCA_factor = 7

    (DTR, LTR), (DTE, LTE) = loadPulsar()
    # Z normalization of data
    DTR, meanDTR, standardDeviationDTR = ZNormalization(DTR)
    DTE, meanDTE, standardDeviationDTE = ZNormalization(DTE)

    if PCA_flag:
        DTR=PCA(DTR,PCA_factor)
        DTE=PCA(DTE,PCA_factor)

    EMStop=1e-6
    Mmax=64   #numero massimo di componenti

    xplot=[2**(i+1) for i in range(int(np.log2(Mmax)))]
    start=time.time()
    dcf1,dcf5,dcf9 = GMMClf(DTR, LTR, Mmax)
    print(time.time()-start)

    plt.figure()
    print('dcf1:', dcf1)
    print('dcf5:', dcf5)
    print('dcf9:', dcf9)
    plt.plot(xplot, dcf1, label='minDCF prior=0.1')
    plt.plot(xplot, dcf5, label='minDCF prior=0.5')
    plt.plot(xplot, dcf9, label='minDCF prior=0.9')
    plt.legend()
    plt.xlabel('components')
    plt.ylabel('min DCF')
    plt.xscale('log')
    plt.savefig('plotNoPCAFull.svg', bbox_inches='tight')
    plt.savefig('plotNoPCAFull.png', bbox_inches='tight')

    #DIAG
    dcf1, dcf5, dcf9 = GMMClf(DTR, LTR, Mmax,diag=1)

    plt.figure()
    print('dcf1:', dcf1)
    print('dcf5:', dcf5)
    print('dcf9:', dcf9)
    plt.plot(xplot, dcf1, label='minDCF prior=0.1')
    plt.plot(xplot, dcf5, label='minDCF prior=0.5')
    plt.plot(xplot, dcf9, label='minDCF prior=0.9')
    plt.legend()
    plt.xlabel('components')
    plt.ylabel('min DCF')
    plt.xscale('log')
    plt.savefig('plotNoPCAdiag.svg', bbox_inches='tight')
    plt.savefig('plotNoPCAdiag.png', bbox_inches='tight')

    #TIED

    dcf1, dcf5, dcf9 = GMMClf(DTR, LTR, Mmax,tied=1)

    plt.figure()
    print('dcf1:', dcf1)
    print('dcf5:', dcf5)
    print('dcf9:', dcf9)
    plt.plot(xplot, dcf1, label='minDCF prior=0.1')
    plt.plot(xplot, dcf5, label='minDCF prior=0.5')
    plt.plot(xplot, dcf9, label='minDCF prior=0.9')
    plt.legend()
    plt.xlabel('components')
    plt.ylabel('min DCF')
    plt.xscale('log')
    plt.savefig('plotNoPCAtied.svg', bbox_inches='tight')
    plt.savefig('plotNoPCAtied.png', bbox_inches='tight')

    DTR = PCA(DTR, PCA_factor)
    DTE = PCA(DTE, PCA_factor)

    xplot = [2 ** (i + 1) for i in range(int(np.log2(Mmax)))]
    dcf1, dcf5, dcf9 = GMMClf(DTR, LTR, Mmax)

    plt.figure()
    print('dcf1:', dcf1)
    print('dcf5:', dcf5)
    print('dcf9:', dcf9)
    plt.plot(xplot, dcf1, label='minDCF prior=0.1')
    plt.plot(xplot, dcf5, label='minDCF prior=0.5')
    plt.plot(xplot, dcf9, label='minDCF prior=0.9')
    plt.legend()
    plt.xlabel('components')
    plt.ylabel('min DCF')
    plt.xscale('log')
    plt.savefig('plotPCA7Full.svg', bbox_inches='tight')
    plt.savefig('plotPCA7Full.png', bbox_inches='tight')

    dcf1, dcf5, dcf9 = GMMClf(DTR, LTR, Mmax,diag=1)

    plt.figure()
    print('dcf1:', dcf1)
    print('dcf5:', dcf5)
    print('dcf9:', dcf9)
    plt.plot(xplot, dcf1, label='minDCF prior=0.1')
    plt.plot(xplot, dcf5, label='minDCF prior=0.5')
    plt.plot(xplot, dcf9, label='minDCF prior=0.9')
    plt.legend()
    plt.xlabel('components')
    plt.ylabel('min DCF')
    plt.xscale('log')
    plt.savefig('plotPCA7diag.svg', bbox_inches='tight')
    plt.savefig('plotPCA7diag.png', bbox_inches='tight')

    dcf1, dcf5, dcf9 = GMMClf(DTR, LTR, Mmax,tied=1)

    plt.figure()
    print('dcf1:', dcf1)
    print('dcf5:', dcf5)
    print('dcf9:', dcf9)
    plt.plot(xplot, dcf1, label='minDCF prior=0.1')
    plt.plot(xplot, dcf5, label='minDCF prior=0.5')
    plt.plot(xplot, dcf9, label='minDCF prior=0.9')
    plt.legend()
    plt.xlabel('components')
    plt.ylabel('min DCF')
    plt.savefig('plotPCA7tied.svg', bbox_inches='tight')
    plt.savefig('plotPCA7tied.png', bbox_inches='tight')
