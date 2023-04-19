import numpy as np
import scipy.special
import json
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size,1))
def vrow(v):
    return v.reshape((1,v.size))
"""
def compute_mu_Sigma(D,L,nClasses):
    D_c = [D[:, L == i] for i in range(nClasses)]
    mu_c = np.array([vcol(D_c[i].mean(1)) for i in range(nClasses)])  # make column vector
    CD_c = [D_c[i] - mu_c[i] for i in range(nClasses)]
    S_c = np.array([np.dot(CD_c[i], CD_c[i].T) / CD_c[i].shape[1] for i in range(nClasses)])
    return mu_c,S_c
"""

def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def splitGMM(GMM_n,alpha):
    GMM_n1 = []
    for g in range(len(GMM_n)):
        U, s, Vh = np.linalg.svd(GMM_n[g][2])  # GMM_1[g][2]
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        GMM_n1.append((GMM_n[g][0] / 2, GMM_n[g][1] + d, GMM_n[g][2]))
        GMM_n1.append((GMM_n[g][0] / 2, GMM_n[g][1] - d, GMM_n[g][2]))
    return GMM_n1

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

def EM_GMMparams(GMMdata,GMMparams,EMStop):
    avgll = 0
    llDiff = 100

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

        GMMparams=[(wnext[i],vcol(munext[i]),Cnext[i]) for i in range(M)]

        newAvgll=logpdf_GMM(GMMdata,GMMparams).sum()/N
        llDiff=np.abs(newAvgll-avgll)
        avgll=newAvgll

        print('log likelihood: ',avgll)

    return GMMparams

def LBGAlgorithm(mu,S,EMStop,alpha,splitnum):
    GMM_n = [(1.0, mu, S)]

    for i in range(splitnum):
        GMM_n = splitGMM(GMM_n, alpha)

        GMM_nEM = EM_GMMparams(GMMdata, GMM_n, EMStop)

    return GMM_nEM

if __name__=='__main__':
    GMMdata = np.load('Data/GMM_data_1D.npy')

    mu = np.array(vcol(GMMdata.mean(1)))
    centerData=GMMdata-mu
    S=np.dot(centerData,centerData.T)/centerData.shape[1]

    solutionLBG=load_gmm('Data/GMM_1D_4G_EM_LBG.json')

    #LBG algorithm
    GMM_1=[(1.0,mu,S)]
    EMStop = 1e-6
    alpha=0.1

    #GMM_4EM=LBGAlgorithm(mu,S,EMStop,alpha,2)

    GMM_2=splitGMM(GMM_1,alpha)

    GMM_2EM = EM_GMMparams(GMMdata, GMM_2, EMStop)

    print('split in 4')
    GMM_4 = splitGMM(GMM_2EM, alpha)

    GMM_4EM=EM_GMMparams(GMMdata,GMM_4,EMStop)

    M=4
    print("error LBG w: ", np.abs([solutionLBG[i][0] - GMM_4EM[-(i+1)][0] for i in range(M)]).max())
    print("error LBG mu: ", np.abs([solutionLBG[i][1] - GMM_4EM[-(i+1)][1] for i in range(M)]).max())
    print("error LBG Sigma: ", np.abs([solutionLBG[i][2] - GMM_4EM[-(i+1)][2] for i in range(M)]).max())

    plt.figure()
    plt.hist(GMMdata.ravel(), bins=50, density=True)
    XPlot = np.linspace(-10, 5, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GMM(vrow(XPlot), GMM_4EM)))
    plt.show()
