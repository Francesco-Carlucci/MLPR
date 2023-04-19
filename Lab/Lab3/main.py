import numpy
import matplotlib.pyplot as plt
import matplotlib

def load():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

if __name__ == '__main__':
    #in pdf c'é D[:,i:i+1]; é uguale a D[:,i] ma già in colonna
    D,L=load()
    print('data: ',D)
    mu=D.mean(1)  #mean on the columns
    print(mu.shape,'mu:\n', mu)
    DC=D-vcol(mu)  #sottrae ad ogni colonna la sua media: [x1-mu1,x2-mu2,...]
    C=numpy.dot(DC,DC.T)/D.shape[1]
    print(C.shape,'c:\n',C)
    s,U=numpy.linalg.eigh(C)
    print('eigenvalues: ',s)
    print('eigenvectors: ', U)

    m=2
    P=U[:,::-1][:,0:m] #eigenvector corresponding to the largest m eigenvalues
    print('first m eigenvectors: ',P)
    #U, s, Vt=numpy.linalg.svd(C) #vedere differenza eigh - svd
    #P=U[:,0:m]
    DP1=numpy.dot(P.T,D[:,L==0]) #samples Setosa projected
    DP2 = numpy.dot(P.T, D[:, L == 1])  # samples Setosa projected
    DP3 = numpy.dot(P.T, D[:, L == 2])  # samples Setosa projected
    print('projected data: ',DP1.shape)

    plt.scatter(DP1[0],DP1[1])
    plt.scatter(DP2[0], DP2[1])
    plt.scatter(DP3[0], DP3[1])
    plt.show()
    """
    DProjList=[]
    for i in range(D.shape[1]):
        DProjList=
    """ #vedere codice da video