import numpy as np
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size,1))
def vrow(v):
    return v.reshape((1,v.size))

def logpdf_GAU_ND(x,mu,C):  #C is sigma,covariance matrix
    _,logdetC=np.linalg.slogdet(C)
    invC=np.linalg.inv(C)
    res=[]
    for i in range(x.shape[1]):
        matres=0.5*np.dot(np.dot((vcol(x[:,i])-mu).T,invC),(vcol(x[:,i])-mu))
        logx=-x.shape[0]/2*np.log(2*np.pi)-logdetC/2-matres
        res.append(logx[0][0])

    """
    invC=np.broadcast_to(invC,(x.shape[1],1))
    res1=np.dot((x-mu),invC)
    res2=np.dot(res1,(x-mu).T)
    res3=logdetC/2-res2/2
    res4=-mu.shape[0]/2*np.log(2*np.pi)-res3
    """

    return np.array(res)

def loglikelihood(XND,m,C):

    return np.sum(logpdf_GAU_ND(XND,m,C))

if __name__=='__main__':
    plt.figure()
    XPlot=np.linspace(-8,12,1000)
    m=np.ones((1,1))*1.0
    C=np.ones((1,1))*2.0
    plt.plot(XPlot.ravel(),np.exp(logpdf_GAU_ND(vrow(XPlot),m,C)))


    solutionGAU=np.load('llGAU.npy')
    gau=logpdf_GAU_ND(vrow(XPlot),m,C)
    print('Errore 1: ',np.abs(solutionGAU-gau).max())

    plt.show()

    XND=np.load('XND.npy')
    mu=np.load('muND.npy')
    C=np.load('CND.npy')
    pdfSol=np.load('llND.npy')
    pdfGau=logpdf_GAU_ND(XND,mu,C)
    print('Errore 2: ',np.abs(pdfSol-pdfGau).max())

    m_ML=vcol(XND.mean(1))
    CD=XND-m_ML
    C_ML=np.dot(CD,CD.T)/XND.shape[1]
    ll=loglikelihood(XND,m_ML,C_ML)
    print(ll)

    X1D = np.load('X1D.npy')
    m_ML1 = vcol(X1D.mean(1))
    CD = X1D - m_ML1
    C_ML1 = np.dot(CD, CD.T) / X1D.shape[1]
    ll1 = loglikelihood(X1D, m_ML1, C_ML1)
    print(m_ML1[0],C_ML1[0])

    plt.figure()
    plt.hist(X1D.ravel(),bins=50, density=True)
    XPlot=np.linspace(-8,12,1000)
    plt.plot(XPlot.ravel(),np.exp(logpdf_GAU_ND(vrow(XPlot),m_ML1,C_ML1)))
    plt.show()
    print(ll1)