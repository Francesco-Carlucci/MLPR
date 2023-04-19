import numpy as np
import scipy.optimize

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

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

"""def primalObjF(w,z,x,C):
    csi=1-np.dot(w,x)*z
    dis=np.vstack([csi,np.zeros(csi.shape[1])]).max(axis=0).sum()
    J=0.5*np.dot(w,w.T)+C*dis
    return J
"""

def optimWeight(C,DTR,LTR,kFunc):
    N=DTR.shape[1]
    x0 = np.zeros(N)

    Z = vrow(2 * LTR - 1)
    #kvet = np.ones(N) * K
    #Dext = np.vstack([DTR, kvet])
    G = kFunc(DTR,DTR)
    H = G * Z * Z.T

    boxcond = [(0, C) for i in range(N)]

    alfaStar, dualLoss, d = scipy.optimize.fmin_l_bfgs_b(lobjFunc, x0, factr=1.0, args=(H,), bounds=boxcond)

    #wstar = np.dot(alfaStar * Z[0], DTR.T)

    return alfaStar,dualLoss

def computeAcc(DTE,LTE,DTR,LTR,alfa,kFunc):
    Z = vcol(2 * LTR - 1)
    S=np.sum(vcol(alfa)*Z*kFunc(DTR,DTE),axis=0)
    predLabels=S>0
    correct=np.sum(predLabels==LTE)
    return correct/LTE.shape[0]

def kPoly(c,d,K):

    def kFunc(x1,x2):
        return (np.dot(x1.T, x2) + c) ** d + np.sqrt(K)

    return kFunc
def kRBF(gamma,K):

    def kFunc(x1,x2):
        res=np.empty((x1.shape[1],x2.shape[1]))
        for i in range(x1.shape[1]):
            for j in range(x2.shape[1]):
                res[i,j]=np.exp(-gamma*np.linalg.norm(x1[:,i]-x2[:,j])**2)
        #res=[np.exp(-gamma*np.linalg.norm(x1-vcol(x2[:,i]))**2) for i in range(x2.shape[1])]
        return res+np.sqrt(K)
    return kFunc

if __name__=="__main__":

    D,L=load_iris_binary()

    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)

    K=0
    C=1.0
    d=2
    c=0
    kernel=kPoly(c,d,K)
    alfaStar,dualLoss=optimWeight(C,DTR,LTR,kernel)
    acc = computeAcc(DTE, LTE, DTR,LTR, alfaStar,kernel)
    print('Dual loss (K=', K, ' C=', C, ') Poly(d=%d,c=%d) : '%(d,c), -dualLoss,'error rate: ',1-acc)

    K = 1.0
    kernel = kPoly(c, d, K)
    alfaStar, dualLoss = optimWeight(C, DTR, LTR, kernel)
    acc = computeAcc(DTE, LTE, DTR, LTR, alfaStar, kernel)
    print('Dual loss (K=', K, ' C=', C, ') Poly(d=%d,c=%d) : '%(d,c), -dualLoss,'error rate: ',1-acc)

    K = 0.0
    c=1
    kernel = kPoly(c, d, K)
    alfaStar, dualLoss = optimWeight(C, DTR, LTR, kernel)
    acc = computeAcc(DTE, LTE, DTR, LTR, alfaStar, kernel)
    print('Dual loss (K=', K, ' C=', C, ') Poly(d=%d,c=%d) : ' % (d, c), -dualLoss, 'error rate: ', 1 - acc)

    K = 1.0
    kernel = kPoly(c, d, K)
    alfaStar, dualLoss = optimWeight(C, DTR, LTR, kernel)
    acc = computeAcc(DTE, LTE, DTR, LTR, alfaStar, kernel)
    print('Dual loss (K=', K, ' C=', C, ') Poly(d=%d,c=%d) : ' % (d, c), -dualLoss, 'error rate: ', 1 - acc)

    K = 0.0
    C = 1.0
    gamma=1.0
    kernel = kRBF(gamma,K)
    alfaStar, dualLoss = optimWeight(C, DTR, LTR, kernel)
    acc = computeAcc(DTE, LTE, DTR, LTR, alfaStar, kernel)
    print('Dual loss (K=', K, ' C=', C, ') RBF(gamma=%d) : ' % gamma, -dualLoss, 'error rate: ', 1 - acc)

    gamma=10.0
    kernel = kRBF(gamma,K)
    alfaStar, dualLoss = optimWeight(C, DTR, LTR, kernel)
    acc = computeAcc(DTE, LTE, DTR, LTR, alfaStar, kernel)
    print('Dual loss (K=', K, ' C=', C, ') RBF(gamma=%d) : ' % gamma, -dualLoss, 'error rate: ', 1 - acc)

    K = 1.0
    gamma=1.0
    kernel = kRBF(gamma,K)
    alfaStar, dualLoss = optimWeight(C, DTR, LTR, kernel)
    acc = computeAcc(DTE, LTE, DTR, LTR, alfaStar, kernel)
    print('Dual loss (K=', K, ' C=', C, ') RBF(gamma=%d) : ' % gamma, -dualLoss, 'error rate: ', 1 - acc)

    K = 1.0
    gamma=10.0
    kernel = kRBF(gamma,K)
    alfaStar, dualLoss = optimWeight(C, DTR, LTR, kernel)
    acc = computeAcc(DTE, LTE, DTR, LTR, alfaStar, kernel)
    print('Dual loss (K=', K, ' C=', C, ') RBF(gamma=%d) : ' % gamma, -dualLoss, 'error rate: ', 1 - acc)
