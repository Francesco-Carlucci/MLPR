import scipy.optimize
import sklearn.datasets
import numpy as np

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

def load_iris_binary():
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

def logreg_obj(v,DTR,LTR,l):

    w,b=v[0:-1],v[-1]
    vsum=0
    for i in range(len(LTR)):
        vsum+=np.logaddexp(0,-(2*LTR[i]-1)*(np.dot(w,DTR[:,i])+b))
    return l/2. * np.dot(w.T,w)+1/len(LTR)*vsum
def computeAcc(DTE,LTE,w,b):
    S=np.dot(w.T,DTE)+b
    predLabels=S>0
    correct=np.sum(predLabels==LTE)
    return correct/LTE.shape[0]

if __name__=='__main__':
    D,L=load_iris_binary()
    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)
    x0=np.zeros(DTR.shape[0]+1)

    l=1.0
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR,LTR,l))
    acc = computeAcc(DTE, LTE, x[:-1], x[-1])
    print('min value (lambda %.6f): '%l,min,'Error rate: ',1-acc)

    l = 0.1
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR, LTR, l))
    acc = computeAcc(DTE, LTE, x[:-1], x[-1])
    print('min value (lambda %.6f): ' % l, min,'Error rate: ',1-acc)

    l=0.001
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR, LTR, l))
    acc = computeAcc(DTE, LTE, x[:-1], x[-1])
    print('min value (lambda %.6f): ' % l, min,'Error rate: ',1-acc)

    l = 1e-6
    x, min, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(DTR, LTR, l))
    acc = computeAcc(DTE, LTE, x[:-1], x[-1])
    print('min value (lambda %.6f): ' % l, min,'Error rate: ',1-acc)

    
