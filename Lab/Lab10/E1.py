import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import json

def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def vcol(v):
    return v.reshape((v.size,1))
def vrow(v):
    return v.reshape((1,v.size))

def logpdf_GAU_ND(x, mu, C):  # C is sigma,covariance matrix
    _, logdetC = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    """
    res = []
    for i in range(x.shape[1]):
        matres = 0.5 * np.dot(np.dot((vcol(x[:, i]) - mu).T, invC), (vcol(x[:, i]) - mu))
        logx = -x.shape[0] / 2 * np.log(2 * np.pi) - logdetC / 2 - matres
        res.append(logx[0][0])
    """
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

if __name__=='__main__':
    #4D
    GMMdata=np.load('Data/GMM_data_4D.npy')
    GMMparams=load_gmm('Data/GMM_4D_3G_init.json')
    resultGMM=np.load('Data/GMM_4D_3G_init_ll.npy')
    llGMM=logpdf_GMM(GMMdata,GMMparams)

    print("GMM error 4D: ",np.abs(resultGMM-llGMM).max())
    #1D
    GMMdata1d = np.load('Data/GMM_data_1D.npy')
    GMMparams1d = load_gmm('Data/GMM_1D_3G_init.json')
    resultGMM1d = np.load('Data/GMM_1D_3G_init_ll.npy')
    llGMM1d = logpdf_GMM(GMMdata1d, GMMparams1d)

    print("GMM error 1D: ", np.abs(resultGMM1d - llGMM1d).max())

    #EXPECTATION MAXIMIZATION algorithm
    EMStop=1e-6
    M = len(GMMparams)

    GMMparamsEM=EM_GMMparams(GMMdata,GMMparams,EMStop)

    solutionEM=load_gmm('Data/GMM_4D_3G_EM.json')

    print("error EM w: ", np.abs([solutionEM[i][0]-GMMparamsEM[i][0] for i in range(M)]).max())
    print("error EM mu: ", np.abs([solutionEM[i][1] - GMMparamsEM[i][1] for i in range(M)]).max())
    print("error EM Sigma: ", np.abs([solutionEM[i][2] - GMMparamsEM[i][2] for i in range(M)]).max())

    GMMparamsEM1d=EM_GMMparams(GMMdata1d,GMMparams1d,EMStop)

    solutionEM1d = load_gmm('Data/GMM_1D_3G_EM.json')

    print("error EM w 1d: ", np.abs([solutionEM1d[i][0] - GMMparamsEM1d[i][0] for i in range(M)]).max())
    print("error EM mu 1d: ", np.abs([solutionEM1d[i][1] - GMMparamsEM1d[i][1] for i in range(M)]).max())
    print("error EM Sigma 1d: ", np.abs([solutionEM1d[i][2] - GMMparamsEM1d[i][2] for i in range(M)]).max())

    plt.figure()
    plt.hist(GMMdata1d[0,: : 2],bins=50,density=True)
    XPlot=np.linspace(-10,5,1000)
    plt.plot(XPlot.ravel(),np.exp(logpdf_GMM(vrow(XPlot), GMMparamsEM1d)))
    plt.show()
