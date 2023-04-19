import numpy as np
import matplotlib.pyplot as plt
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
def computePost(DTE,mu_c,S_c,nClasses=3):
    # matrix of the likelihoods f(x|c)
    S = np.array([np.exp(logpdf_GAU_ND(DTE, mu_c[i], S_c[i])) for i in range(nClasses)])  # scores
    # matrix of joint densities f(x,c), multiply for the class prior probability, 1/3
    SJoint = S * [[1.0 / 3.0]]
    # compute the marginal probability, f(x)
    SMarginal = vrow(SJoint.sum(0))
    # class posterior probability P(c|x)
    SPost = SJoint / SMarginal  # P(c|x)=f(x,c)/f(x)
    return SPost,SJoint,SMarginal

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

def printAccuracy(predLabels,LTE):
    correct = np.sum(predLabels == LTE)
    accuracy = correct / LTE.shape[0]
    errRate = 1 - accuracy
    print('accuracy: ', accuracy)
    print('error rate: %.3f' % errRate)

def calculateAccfromPost(SPost,LTE):
    predLabels = np.argmax(SPost, axis=0)
    printAccuracy(predLabels,LTE)

if __name__=='__main__':
    D,L=load()
    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)
    # check result
    solutionSJoint = np.load('SJoint_MVG.npy')
    solutionSPost = np.load('Posterior_MVG.npy')

    solutionLogSMarginal = np.load('logMarginal_MVG.npy')
    solutionLogSPost = np.load('logPosterior_MVG.npy')
    solutionLogSJoint=np.load('logSJoint_MVG.npy')

    solutionNaiveSPost=np.load('Posterior_NaiveBayes.npy')
    solutionNaiveSJoint=np.load('SJoint_NaiveBayes.npy')

    solutionNaiveLogSPost=np.load('logPosterior_NaiveBayes.npy')
    solutionNaiveLogSJoint=np.load('logSJoint_NaiveBayes.npy')
    solutionNaiveLogSMarginal=np.load('logMarginal_NaiveBayes.npy')

    solutionTiedSPost=np.load('Posterior_TiedMVG.npy')
    solutionTiedSJoint=np.load('SJoint_TiedMVG.npy')

    solutionTiedLogSPost=np.load('logPosterior_TiedMVG.npy')
    solutionTiedLogSJoint=np.load('logSJoint_TiedMVG.npy')
    solutionTiedLogSMarginal=np.load('logMarginal_TiedMVG.npy')

    solutionTiedNaiveSPost=np.load('Posterior_TiedNaiveBayes.npy')
    solutionTiedNaiveSJoint=np.load('SJoint_TiedNaiveBayes.npy')

    solutionTiedNaiveLogSPost=np.load('logPosterior_TiedNaiveBayes.npy')
    solutionTiedNaiveLogSJoint=np.load('logSJoint_TiedNaiveBayes.npy')
    solutionTiedNaiveLogSMarginal=np.load('logMarginal_TiedNaiveBayes.npy')

    nClasses=3
    #MVG classifier, estimate parameters
    mu_c,S_c=compute_mu_Sigma(DTR,LTR,nClasses)

    print('μ 0: ',mu_c[0], '\nΣ 0: ',S_c[0])
    print('μ 1: ', mu_c[1], '\nΣ 1: ', S_c[1])
    print('μ 2: ', mu_c[2], '\nΣ 2: ', S_c[2])

        #INFERENCE, compute the log-likelihoods
    print('                    MVG')
    SPost,SJoint,_=computePost(DTE,mu_c,S_c,nClasses)
    print('joint densities error: ', np.abs(SJoint - solutionSJoint).max())
    print('post probabilities error: ',np.abs(SPost-solutionSPost).max())

    #calculate accuracy
    calculateAccfromPost(SPost,LTE)

    #PROCESS WITH ONLY LOG-DENSITIES
    print('                   logMVG')
    logSPost,logSJoint,logSMarginal=computeLogPost(DTE,mu_c,S_c,nClasses)

    print('log post error: ',np.abs(logSPost - solutionLogSPost).max())
    print('log joint error: ',np.abs(logSJoint-solutionLogSJoint).max())
    print('log marginal error: ',np.abs(logSMarginal-solutionLogSMarginal).max())

    #NAIVE BAYES
    print('                   NAIVE BAYES')
    identity=np.identity(S_c.shape[1])
    naiveS_c=[S_c[i]*identity for i in range(nClasses)]
    naiveSPost,naiveSJoint,naiveSMarginal=computePost(DTE,mu_c,naiveS_c,nClasses)

    print('naive post error: ', np.abs(naiveSPost - solutionNaiveSPost).max())
    print('naive joint error: ', np.abs(naiveSJoint - solutionNaiveSJoint).max())
    #print('naive marginal error: ', np.abs(naiveSMarginal - solutionNaiveSMarginal).max())

    naiveLogSPost,naiveLogSJoint,naiveLogSMarginal=computeLogPost(DTE,mu_c,naiveS_c)

    print('naive log post error: ', np.abs(naiveLogSPost - solutionNaiveLogSPost).max())
    print('naive log joint error: ', np.abs(naiveLogSJoint - solutionNaiveLogSJoint).max())
    print('naive log marginal error: ', np.abs(naiveLogSMarginal - solutionNaiveLogSMarginal).max())

    calculateAccfromPost(naiveLogSPost,LTE)

    #TIED MVG
    print('                  TIED MVG')
    tiedS_c=np.sum([S_c[i]*np.sum(LTR==i) for i in range(nClasses)],axis=0)/DTR.shape[1]
    print('tied covariance matrix:\n', tiedS_c)
    tiedS_c=[tiedS_c for i in range(nClasses)]
    tiedSPost,tiedSJoint,tiedSMarginal=computePost(DTE,mu_c,tiedS_c,nClasses)

    print('tied post error: ', np.abs(tiedSPost - solutionTiedSPost).max())
    print('tied joint error: ', np.abs(tiedSJoint - solutionTiedSJoint).max())

    tiedLogSPost,tiedLogSJoint,tiedLogSMarginal=computeLogPost(DTE,mu_c,tiedS_c,nClasses)

    print('tied log post error: ', np.abs(tiedLogSPost - solutionTiedLogSPost).max())
    print('tied log joint error: ', np.abs(tiedLogSJoint - solutionTiedLogSJoint).max())
    print('tied log marginal error: ', np.abs(tiedLogSMarginal - solutionTiedLogSMarginal).max())

    calculateAccfromPost(tiedSPost,LTE)
    print('              TIED NAIVE BAYES')
    tiedNaiveS_c=[tiedS_c[i]*identity for i in range(nClasses)]
    tiedNaiveSPost,tiedNaiveSJoint,tiedNaiveSMarginal=computePost(DTE,mu_c,tiedNaiveS_c,nClasses)

    print('tied naive post error: ', np.abs(tiedNaiveSPost - solutionTiedNaiveSPost).max())
    print('tied naive joint error: ', np.abs(tiedNaiveSJoint - solutionTiedNaiveSJoint).max())

    tiedNaiveLogSPost, tiedNaiveLogSJoint, tiedNaiveLogSMarginal = computeLogPost(DTE, mu_c, tiedNaiveS_c, nClasses)

    print('tied log post error: ', np.abs(tiedNaiveLogSPost - solutionTiedNaiveLogSPost).max())
    print('tied log joint error: ', np.abs(tiedNaiveLogSJoint - solutionTiedNaiveLogSJoint).max())
    print('tied log marginal error: ', np.abs(tiedNaiveLogSMarginal - solutionTiedNaiveLogSMarginal).max())

    calculateAccfromPost(tiedNaiveSPost,LTE)

    print('                 K-fold')
    solSJointMVG=np.load('LOO_logSJoint_MVG.npy')
    solSJointNB = np.load('LOO_logSJoint_NaiveBayes.npy')
    solSJointTied = np.load('LOO_logSJoint_TiedMVG.npy')
    solSJointTiedNaive = np.load('LOO_logSJoint_TiedNaiveBayes.npy')

    predLabelsMVG=[]
    predLabelsNB=[]  #Naive Bayes
    predLabelsTied=[]
    predLabelsTiedNaive=[]
    for i in range(D.shape[1]):
        #idx=range(D.shape[1])
        if i!=0:
            DTR=np.hstack([D[:,:i:],D[:,i+1::]])
            LTR=np.hstack([L[:i:],L[i+1::]])
        else:
            DTR=D[:,i+1::]
            LTR=L[i+1::]
        mu_c, S_c = compute_mu_Sigma(DTR, LTR, nClasses)
        logMVGSPost, logMVGSJoint, logMVGSMarginal = computeLogPost(vcol(D[:,i]), mu_c, S_c, nClasses)
        predLabelsMVG.append(np.argmax(np.exp(logMVGSPost), axis=0)[0])
        #naive bayes
        identity = np.identity(S_c.shape[1])
        naiveS_c = [S_c[i] * identity for i in range(nClasses)]
        logNBSPost, logNBSJoint, _ = computeLogPost(vcol(D[:, i]), mu_c, naiveS_c, nClasses)
        predLabelsNB.append(np.argmax(np.exp(logNBSPost), axis=0)[0])
        #tied
        tiedS_c = np.sum([S_c[i] * np.sum(LTR == i) for i in range(nClasses)], axis=0) / DTR.shape[1]
        tiedS_c = [tiedS_c for i in range(nClasses)]
        logTiedSPost,logTiedSJoint,_=computeLogPost(vcol(D[:, i]), mu_c, tiedS_c, nClasses)
        predLabelsTied.append(np.argmax(np.exp(logTiedSPost), axis=0)[0])
        #tied Naive
        tiedNaiveS_c = [tiedS_c[i] * identity for i in range(nClasses)]
        logTiedNaiveSPost, logTiedNaiveSJoint, _ = computeLogPost(vcol(D[:, i]), mu_c, tiedNaiveS_c, nClasses)
        predLabelsTiedNaive.append(np.argmax(np.exp(logTiedNaiveSPost), axis=0)[0])

        print('MVG log joint error: ', np.abs(logMVGSJoint - solSJointMVG).max())
        print('NAIVE log joint error: ', np.abs(logNBSJoint - solSJointNB).max())
        print('TIED log joint error: ', np.abs(logTiedSJoint - solSJointTied).max())
        print('TIED NAIVE log joint error: ', np.abs(logTiedNaiveSJoint - solSJointTiedNaive).max())

    print('MVG')
    printAccuracy(predLabelsMVG,L)
    print('NAIVE BAYES')
    printAccuracy(predLabelsNB,L)
    print('TIED MVG')
    printAccuracy(predLabelsTied,L)
    print('TIED NAIVE BAYES')
    printAccuracy(predLabelsTiedNaive,L)

