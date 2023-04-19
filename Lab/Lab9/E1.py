import numpy as np
import scipy.optimize

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

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

def gradL(alfa,H):
    I = np.identity(H.shape[0])
    grad=np.dot(H,alfa)-I

    return grad

def lobjFunc(alfa,H):
    N=H.shape[0]
    I=vcol(np.ones(N))
    alfa=vcol(alfa)
    L=0.5*np.dot(np.dot(alfa.T,H),alfa)-np.dot(alfa.T,I)

    grad = np.dot(H, alfa) - I

    return L,grad.reshape(N)

def primalObjF(w,z,x,C):
    csi=1-np.dot(w,x)*z
    dis=np.vstack([csi,np.zeros(csi.shape[1])]).max(axis=0).sum()
    J=0.5*np.dot(w,w.T)+C*dis
    return J

def optimWeight(K,C,DTR,LTR):
    N=DTR.shape[1]
    x0 = np.zeros(N)

    Z = vrow(2 * LTR - 1)
    kvet = np.ones(N) * K
    Dext = np.vstack([DTR, kvet])
    G = np.dot(Dext.T, Dext)
    H = G * Z * Z.T

    boxcond = [(0, C) for i in range(N)]

    alfaStar, dualLoss, d = scipy.optimize.fmin_l_bfgs_b(lobjFunc, x0, factr=1.0, args=(H,), bounds=boxcond)

    wstar = np.dot(alfaStar * Z[0], Dext.T)

    primalLoss = primalObjF(wstar, Z, Dext,C)

    return alfaStar,wstar,dualLoss,primalLoss

def computeAcc(DTE,LTE,w,b,K):
    S=np.dot(w.T,DTE)+b*K
    predLabels=S>0
    correct=np.sum(predLabels==LTE)
    return correct/LTE.shape[0]

if __name__=="__main__":

    D,L=load_iris_binary()

    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)

    K=1
    C=0.1
    _,w,dualLoss,primalLoss=optimWeight(K,C,DTR,LTR)
    dualGap = primalLoss + dualLoss
    acc = computeAcc(DTE, LTE, w[:-1], w[-1],K)
    print('Dual loss (K=', K, ' C=', C, ') : ', -dualLoss,' Primal loss: ',primalLoss,' duality gap: ',dualGap,'error rate: ',1-acc)

    C=1.0
    _, w, dualLoss, primalLoss = optimWeight(K,C,DTR, LTR)
    dualGap = primalLoss + dualLoss
    acc = computeAcc(DTE, LTE, w[:-1], w[-1],K)
    print('Dual loss (K=', K, ' C=', C, ') : ', -dualLoss,' Primal loss: ',primalLoss,' duality gap: ',dualGap,'error rate: ',1-acc)

    C=10.0
    _, w, dualLoss, primalLoss = optimWeight(K,C, DTR, LTR)
    dualGap=primalLoss+dualLoss
    acc = computeAcc(DTE, LTE, w[:-1], w[-1],K)
    print('Dual loss (K=', K, ' C=', C, ') : ', -dualLoss,' Primal loss: ',primalLoss,' duality gap: ',dualGap,'error rate: ',1-acc)

    K=10
    C=0.1
    _, w, dualLoss, primalLoss = optimWeight(K,C, DTR, LTR)
    dualGap = primalLoss + dualLoss
    acc = computeAcc(DTE, LTE, w[:-1], w[-1],K)
    print('Dual loss (K=', K, ' C=', C, ') : ', -dualLoss, ' Primal loss: ', primalLoss, ' duality gap: ', dualGap,
          'error rate: ', 1 - acc)
    C=1.0
    _, w, dualLoss, primalLoss = optimWeight(K,C, DTR, LTR)
    dualGap = primalLoss + dualLoss
    acc = computeAcc(DTE, LTE, w[:-1], w[-1],K)
    print('Dual loss (K=', K, ' C=', C, ') : ', -dualLoss, ' Primal loss: ', primalLoss, ' duality gap: ', dualGap,
          'error rate: ', 1 - acc)

    C=10.0
    _, w, dualLoss, primalLoss = optimWeight(K,C, DTR, LTR)
    dualGap = primalLoss + dualLoss
    acc = computeAcc(DTE, LTE, w[:-1], w[-1],K)
    print('Dual loss (K=', K, ' C=', C, ') : ', -dualLoss, ' Primal loss: ', primalLoss, ' duality gap: ', dualGap,
          'error rate: ', 1 - acc)









