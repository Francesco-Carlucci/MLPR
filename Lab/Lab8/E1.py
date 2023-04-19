import numpy as np
import scipy
from scipy import special

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

def loadIris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

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

def compute_mu_Sigma(D,L,nClasses):
    D_c = [D[:, L == i] for i in range(nClasses)]
    mu_c = np.array([vcol(D_c[i].mean(1)) for i in range(nClasses)])  # make column vector
    CD_c = [D_c[i] - mu_c[i] for i in range(nClasses)]
    S_c = np.array([np.dot(CD_c[i], CD_c[i].T) / CD_c[i].shape[1] for i in range(nClasses)])
    return mu_c,S_c

def logpdf_GAU_ND(x, mu, C):  # C is sigma,covariance matrix
        _, logdetC = np.linalg.slogdet(C)
        invC = np.linalg.inv(C)
        res = []
        for i in range(x.shape[1]):
            matres = 0.5 * np.dot(np.dot((vcol(x[:, i]) - mu).T, invC), (vcol(x[:, i]) - mu))
            logx = -x.shape[0] / 2 * np.log(2 * np.pi) - logdetC / 2 - matres
            res.append(logx[0][0])

        return np.array(res)

def computeLogPost(DTE,mu_c,S_c,nClasses=3):
    # matrix of the log-likelihood log(f(x|c))
    logS = np.array([logpdf_GAU_ND(DTE, mu_c[i], S_c[i]) for i in range(nClasses)])  # scores
    # calculate joint-log density
    logSJoint = logS + [[np.log(1.0 / 3.0)]]
    # marginal log-densities
    #l=logSJoint.max(0)
    #logSMarginal1=l+np.log(np.exp(logSJoint-l).sum(0))
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    return logSPost,logSJoint,logSMarginal

def confMat(predLabels,LTE,nClasses):
    confMat = np.zeros((nClasses, nClasses))
    #preCor = [(predLabels[i], LTE[i]) for i in range(predLabels.shape[0])]
    for i in range(nClasses):
        predH = predLabels == i
        for j in range(nClasses):
            corrH = LTE == j
            confMat[i, j] = np.dot(predH.astype(int), corrH.astype(int).T)
    return confMat


import sys


def load_Dante():
    lInf = []

    f = open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f = open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f = open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_Dante(l, n):
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])

    return lTrain, lTest

def computeDictionary(lTrain,nClasses):
    D = {}
    cnt = 0
    for i in range(nClasses):
        for tercet in lTrain[i]:
            for word in tercet.split():
                if word not in D.keys():
                    D[word] = cnt
                    cnt += 1
    return D

def computeOccs(D,text,eps):
    occCnt =np.zeros(len(D.keys()))+eps
    for tercet in text:
        for word in tercet.split():
            if word in D.keys():
                occCnt[D[word]] += 1
    return occCnt

def testOcc(lTest):
    y=[]
    for tercet in lTest:
        y.append(vcol(computeOccs(D, [tercet],0)))

    return np.hstack(y)

if __name__=='__main__':
    D, L = loadIris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    nClasses = 3
    # MVG classifier, estimate parameters
    mu_c, S_c = compute_mu_Sigma(DTR, LTR, nClasses)

    logSPost, logSJoint, _ = computeLogPost(DTE, mu_c, S_c, nClasses)
    predLabels=np.argmax(logSPost,axis=0)
    mvgConfMat=confMat(predLabels,LTE,nClasses)

    print('Confusion matrix: ',mvgConfMat)

    tiedS_c = np.sum([S_c[i] * np.sum(LTR == i) for i in range(nClasses)], axis=0) / DTR.shape[1]
    tiedS_c = [tiedS_c for i in range(nClasses)]
    tiedLogSPost, tiedLogSJoint, tiedLogSMarginal = computeLogPost(DTE, mu_c, tiedS_c, nClasses)
    predLabels=np.argmax(tiedLogSPost,axis=0)

    tiedConfMat = confMat(predLabels, LTE, nClasses)

    print('Tied Confusion matrix: ', tiedConfMat)

    eps=0.001
    lInf, lPur, lPar = load_Dante()

    lInf_train, lInf_evaluation = split_Dante(lInf, 4)
    lPur_train, lPur_evaluation = split_Dante(lPur, 4)
    lPar_train, lPar_evaluation = split_Dante(lPar, 4)

    lTrain = [lInf_train, lPur_train, lPar_train]
    N_c = [len(lTrain[i]) for i in range(nClasses)]

    D = computeDictionary(lTrain, nClasses)
    occCnt = []
    for i in range(nClasses):
        occCnt.append(computeOccs(D, lTrain[i], eps))

    normOcc = [occCnt[i] / occCnt[i].sum() for i in range(nClasses)]
    w_c = np.log(normOcc)

    y = testOcc(lInf_evaluation+lPur_evaluation+lPar_evaluation)
    Sinf = np.array([vrow(np.dot(y.T, vcol(w_c[i])))[0] for i in range(nClasses)])
    SJoint = Sinf + np.log([[1.0 / 3.0]])
    SMarginal = special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SMarginal

    predLabels=np.argmax(SPost,axis=0)
    corrLabels=np.concatenate((np.zeros(len(lInf_evaluation)),np.ones(len(lPur_evaluation)),np.ones(len(lPar_evaluation))*2))
    mvgConfMat = confMat(predLabels, corrLabels, nClasses)

    print('Dante confusion matrix: ',mvgConfMat)