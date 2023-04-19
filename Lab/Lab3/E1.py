import numpy as np
import matplotlib.pyplot as plt
import scipy

def load():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

def computePCA(D,m):
    #calculate the mean
    mu=vcol(D.mean(1))
    #center data
    CD=D-mu
    #calculate covariance matrix
    C=np.dot(CD,CD.T)/D.shape[1]
    #calculate eigenvectors and the eigenvalues, sorted
    s, U=np.linalg.eigh(C)
    #eig does not sort!!!!
    P=U[:,::-1][:,0:m] #reverse and select first m rows

    #U,s,Vh=np.linalg.svd(C)
    #P=U[:,0:m] #already sorted descending

    #projecting the dataset
    plt.figure()
    plt.title('PCA')
    PD0=np.dot(P.T,D[:,L==0])
    plt.scatter(PD0[0,:],PD0[1,:],label='Setosa')
    PD1 = np.dot(P.T, D[:, L == 1])
    plt.scatter(PD1[0, :], PD1[1, :],label='Versicolor')
    PD2 = np.dot(P.T, D[:, L == 2])
    plt.scatter(PD2[0, :], PD2[1, :],label='Virginica')
    plt.legend()

def computeLDA(D,L,m):
    nClasses=3
    mu = vcol(D.mean(1)) #media di tutto il dataset
    D0=D[:,L==0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]
    """
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))
    mu2 = vcol(D2.mean(1))
    CD0 = D0 - vcol(D0.mean(1))
    CD1 = D1 - vcol(D1.mean(1))
    CD2 = D2 - vcol(D2.mean(1))
    Sw = (np.dot(CD0, CD0.T) +
          np.dot(CD1, CD1.T) +
          np.dot(CD2, CD2.T)) / D.shape[1]
    print('Sw: ',Sw)
    cmu0 = mu0 - mu
    cmu1 = mu1 - mu
    cmu2 = mu2 - mu
    Sb = (D0.shape[1] * np.dot(cmu0, cmu0.T) +
          D1.shape[1] * np.dot(cmu1, cmu1.T) +
          D2.shape[1] * np.dot(cmu2, cmu2.T)) / D.shape[1]
    print('Sb: ', Sb)
    """
    D_c=np.array([D[:, L==i] for i in range(nClasses)])
    mu_c=D_c.mean(2).reshape(D_c.shape[0],D_c.shape[1],1)
    CD_c=D_c-mu_c
    Sw=np.sum([np.dot(CD_c[i],CD_c[i].T) for i in range(nClasses)],axis=0)/D.shape[1]

    print('Sw: ',Sw)

    cmu_c=mu_c-mu
    Sb=np.sum([D_c[i].shape[1]*np.dot(cmu_c[i],cmu_c[i].T) for i in range(nClasses)],axis=0)/D.shape[1]

    print('Sb: ', Sb)

    s,U=scipy.linalg.eigh(Sb,Sw)
    W=U[:,::-1][:,0:m] #reverse, take the m highest eigenvectors
    print('W: ',W)
    #UW,_,_=np.linalg.svd(W) #find orthogonal base
    #W=UW[:,0:m]
    #print('W2: ', W)

    #projecting the dataset:
    plt.figure()
    plt.title('LDA')
    #PD = [np.dot(W.T, D_c[i]) for i in range(nClasses)]
    #[plt.scatter(PD[i][0, :], PD[i][1, :]) for i in range(nClasses)]
    PD0=np.dot(W.T,D0)
    plt.scatter(PD0[0,:],PD0[1,:])
    PD1 = np.dot(W.T, D1)
    plt.scatter(PD1[0, :], PD1[1, :])
    PD2 = np.dot(W.T, D2)
    plt.scatter(PD2[0, :], PD2[1, :])

    plt.legend(['Setosa', 'Versicolor', 'Virginica'])


if __name__=='__main__':
    solutionPCA=np.load('IRIS_PCA_matrix_m4.npy')
    solutionLDA = np.load('IRIS_LDA_matrix_m2.npy')
    print('solution PCA: ',solutionPCA,'\nsolution LDA: ',solutionLDA)

    m=4
    D,L=load()

    # print dataset
    plt.figure()
    plt.title('Dataset unreduced')
    plt.scatter(D[0,L==0], D[1,L==0])
    plt.scatter(D[0,L==1], D[1,L==1])
    plt.scatter(D[0,L==2], D[1,L==2])

    computePCA(D,m)

    computeLDA(D, L, 2)
    plt.show()