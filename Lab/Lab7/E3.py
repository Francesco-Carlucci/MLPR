import numpy as np
import scipy


def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

def load():
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
def logreg_obj(v,DTR,LTR,l):
    K = max(LTR)+1
    n = len(LTR)

    w,b=v[0:-K],v[-K:]
    w=w.reshape((DTR.shape[0],-1))
    b=vcol(b)
    T=np.zeros((K,n))
    for i in range(n):
        T[LTR[i],i]=1

    S=np.dot(w.T,DTR)+b
    sumexp=np.exp(S).sum(0)
    ylog=S-np.log(sumexp)

    wNorm=(w*w).sum()

    J=l/2*wNorm-1/n*np.sum(T*ylog)

    return J

def computeAcc(DTE,LTE,w,b):
    S=np.dot(w.T,DTE)+b
    predLabels=np.argmax(S, axis=0)
    correct=np.sum(predLabels==LTE)
    return correct/LTE.shape[0]


if __name__=='__main__':

    D,L=load()
    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)
    K=3

    x0=np.zeros((DTR.shape[0]+1)*K)

    l = 1.0
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR, LTR, l))
    w, b = x[0:-K], x[-K:]
    w = w.reshape((DTR.shape[0], -1))
    b = vcol(b)
    acc = computeAcc(DTE, LTE, w, b)
    print('min value (lambda %.6f): ' % l, min,'Error rate: ',1-acc)

    l = 0.1
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR, LTR, l))
    w, b = x[0:-K], x[-K:]
    w = w.reshape((DTR.shape[0], -1))
    b = vcol(b)
    acc = computeAcc(DTE, LTE, w, b)
    print('min value (lambda %.6f): ' % l, min,'Error rate: ',1-acc)

    l = 1e-3
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR, LTR, l))
    w, b = x[0:-K], x[-K:]
    w = w.reshape((DTR.shape[0], -1))
    b = vcol(b)
    acc = computeAcc(DTE, LTE, w, b)
    print('min value (lambda %.6f): ' % l, min,'Error rate: ',1-acc)

    l = 1e-6
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR, LTR, l))
    w, b = x[0:-K], x[-K:]
    w = w.reshape((DTR.shape[0], -1))
    b = vcol(b)
    acc = computeAcc(DTE, LTE, w, b)
    print('min value (lambda %.6f): ' % l, min,'Error rate: ',1-acc)
