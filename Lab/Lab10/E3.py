import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import json

def vcol(v):
    return v.reshape((v.size,1))
def vrow(v):
    return v.reshape((1,v.size))

def loadIris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def load_iris_binary():
    import sklearn.datasets

    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D,L,seed=0):
    nTrain=int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx=np.random.permutation(D.shape[1])
    idxTrain=idx[0:nTrain]
    idxTest=idx[nTrain:]

    DTR=D[:,idxTrain]  #train_dataset
    DTE=D[:,idxTest]
    LTR=L[idxTrain]
    LTE=L[idxTest]

    return (DTR,LTR),(DTE,LTE)

def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def logpdf_GAU_ND(x, mu, C):  # C is sigma,covariance matrix
    _, logdetC = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    D=x.shape[0]
    Mat=0.5*np.dot(np.dot((x-mu).T,invC),(x-mu)).diagonal()
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

        #print('log likelihood: ',avgll)

    return GMMparams
def splitGMM(GMM_n,alpha):
    GMM_n1 = []
    for g in range(len(GMM_n)):
        U, s, Vh = np.linalg.svd(GMM_n[g][2])  # GMM_1[g][2]
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        GMM_n1.append((GMM_n[g][0] / 2, GMM_n[g][1] + d, GMM_n[g][2]))
        GMM_n1.append((GMM_n[g][0] / 2, GMM_n[g][1] - d, GMM_n[g][2]))
    return GMM_n1

def compute_mu_s(GMMdata):
    mu = np.array(vcol(GMMdata.mean(1)))
    centerData = GMMdata - mu
    S = np.dot(centerData, centerData.T) / centerData.shape[1]

    return mu, S

def constrainCovMat(C,M,psi):
    for g in range(M):
        U, s, _ = np.linalg.svd(C[g])
        s[s < psi] = psi
        C[g] = np.dot(U, vcol(s) * U.T)
    return C

def LBGAlgorithm(GMMdata,mu,S,EMStop,alpha,splitnum,diag,tied):
    GMM_n = EM_GMMparams(GMMdata, [(1.0, mu, S)], EMStop)

    for i in range(splitnum):
        GMM_n = splitGMM(GMM_n, alpha)

        GMM_n = EM_GMMparams(GMMdata, GMM_n, EMStop,diag,tied)  #diag,tied flags

    return GMM_n

def GMMClf(DTR,LTR,M,EMStop=1e-6,alpha=0.1,eps=0.01, diag=0,tied=0 ):

    GMMdata0=DTR[:,LTR==0]
    GMMdata1=DTR[:,LTR==1]
    GMMdata2=DTR[:,LTR==2]

    mu0,S0=compute_mu_s(GMMdata0)
    GMMinit0 = [(1.0, mu0, constrainCovMat([S0],1,eps)[0]) for i in range(M)]
    mu1, S1= compute_mu_s(GMMdata1)
    GMMinit1 = [(1.0, mu1, constrainCovMat([S1],1,eps)[0]) for i in range(M)]
    mu2, S2 = compute_mu_s(GMMdata2)
    GMMinit2 = [(1.0, mu2, constrainCovMat([S2],1,eps)[0]) for i in range(M)]
    """
    GMMparams0 = EM_GMMparams(GMMdata0, GMMinit0, EMStop, diag, tied)
    GMMparams1 = EM_GMMparams(GMMdata1, GMMinit1, EMStop, diag, tied)
    GMMparams2 = EM_GMMparams(GMMdata2, GMMinit2, EMStop, diag, tied)
    """
    splitnum=int(np.log2(M))
    GMMparams0 = LBGAlgorithm(GMMdata0,mu0,S0,EMStop,alpha,splitnum,diag,tied)
    GMMparams1 = LBGAlgorithm(GMMdata1,mu1, S1, EMStop, alpha, splitnum,diag,tied)
    GMMparams2 = LBGAlgorithm(GMMdata2,mu1, S1, EMStop, alpha, splitnum,diag,tied)

    return GMMparams0,GMMparams1,GMMparams2

def predictScores(DTE,GMMparams0,GMMparams1,GMMparams2):

    ll0=logpdf_GMM(DTE,GMMparams0) #compute marginals of each class
    ll1=logpdf_GMM(DTE, GMMparams1)
    ll2=logpdf_GMM(DTE, GMMparams2)

    #llr=ll0-ll1

    return np.vstack([ll0,ll1,ll2])

def computeAcc(scores,LTE):
    predLabels=np.argmax(scores,axis=0)
    correct = np.sum(predLabels == LTE)
    return correct / LTE.shape[0]

if __name__=='__main__':
    GMMdata = np.load('Data/GMM_data_4D.npy')
    GMMparams = load_gmm('Data/GMM_4D_3G_init.json')

    EMStop = 1e-6
    M = len(GMMparams)

    GMMparamsEM = EM_GMMparams(GMMdata, GMMparams, EMStop,0,0)

    solutionEM = load_gmm('Data/GMM_4D_3G_EM.json')

    print("error EM w: ", np.abs([solutionEM[i][0] - GMMparamsEM[i][0] for i in range(M)]).max())
    print("error EM mu: ", np.abs([solutionEM[i][1] - GMMparamsEM[i][1] for i in range(M)]).max())
    print("error EM Sigma: ", np.abs([solutionEM[i][2] - GMMparamsEM[i][2] for i in range(M)]).max())

    D, L = loadIris()

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    GMMparams0,GMMparams1,GMMparams2=GMMClf(DTR, LTR, 1)
    scores=predictScores(DTE,GMMparams0,GMMparams1,GMMparams2)
    accFull1=computeAcc(scores,LTE)
    print("Accuracy full 1: ",accFull1,'error rate: ',1-accFull1)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 2)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accFull2 = computeAcc(scores, LTE)
    print("Accuracy full 2: ", accFull2, 'error rate: ', 1 - accFull2)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 4)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accFull4 = computeAcc(scores, LTE)
    print("Accuracy full 4: ", accFull4, 'error rate: ', 1 - accFull4)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 8)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accFull8 = computeAcc(scores, LTE)
    print("Accuracy full 8: ", accFull8, 'error rate: ', 1 - accFull8)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 16)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accFull16 = computeAcc(scores, LTE)
    print("Accuracy full 16: ", accFull16, 'error rate: ', 1 - accFull16)

    # DIAGONAL MODEL
    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 1,diag=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accDiag1 = computeAcc(scores, LTE)
    print("Accuracy diag 1: ", accDiag1, 'error rate: ', 1 - accDiag1)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 2,diag=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accDiag2 = computeAcc(scores, LTE)
    print("Accuracy diag 2: ", accDiag2, 'error rate: ', 1 - accDiag2)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 4,diag=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accDiag4 = computeAcc(scores, LTE)
    print("Accuracy diag 4: ", accDiag4, 'error rate: ', 1 - accDiag4)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 8,diag=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accDiag8 = computeAcc(scores, LTE)
    print("Accuracy diag 8: ", accDiag8, 'error rate: ', 1 - accDiag8)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 16,diag=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    accDiag16 = computeAcc(scores, LTE)
    print("Accuracy diag 16: ", accDiag16, 'error rate: ', 1 - accDiag16)

    #TIED MODEL
    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 1, tied=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    acctied1 = computeAcc(scores, LTE)
    print("Accuracy tied 1: ", acctied1, 'error rate: ', 1 - acctied1)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 2, tied=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    acctied2 = computeAcc(scores, LTE)
    print("Accuracy tied 2: ", acctied2, 'error rate: ', 1 - acctied2)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 4, tied=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    acctied4 = computeAcc(scores, LTE)
    print("Accuracy tied 4: ", acctied4, 'error rate: ', 1 - acctied4)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 8, tied=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    acctied8 = computeAcc(scores, LTE)
    print("Accuracy tied 8: ", acctied8, 'error rate: ', 1 - acctied8)

    GMMparams0, GMMparams1, GMMparams2 = GMMClf(DTR, LTR, 16, tied=1)
    scores = predictScores(DTE, GMMparams0, GMMparams1, GMMparams2)
    acctied16 = computeAcc(scores, LTE)
    print("Accuracy tied 16: ", acctied16, 'error rate: ', 1 - acctied16)