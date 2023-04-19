import numpy as np
import matplotlib.pyplot as plt
from util import loadPulsar,PCA,ZNormalization,Ksplit
from GMM import logpdf_GAU_ND,constrainCovMat,compute_mu_s,computellr,LBGAlgorithm
import scipy.special

def covarianceMatrix(D):
    mu = D.mean(1)

    DC = D - mu.reshape((D.shape[0], 1))

    C = np.dot(DC, DC.T) / float(DC.shape[1])

    return C


def logpdf_GAU_ND_single(X, mu, C):
    M = mu.shape[0]

    logpdf = -(M / 2.0) * np.log(2.0 * np.pi) - (1 / 2.0) * np.linalg.slogdet(C)[1] - (1 / 2.0) * (
        np.dot(np.dot((X - mu).T, np.linalg.inv(C)), (X - mu)))

    logpdf = logpdf[0][0]

    return logpdf

def logpdf_GAU_ND(X, mu, C):
    logpdf = []

    for i in range(X.shape[-1]):
        xn = X[..., i]

        xn = vcol(xn)
        mu = vcol(mu)

        logpdf.append(logpdf_GAU_ND_single(xn, mu, C))

    return logpdf

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

def predictllr(DTR,LTR,DTE):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    muTR0 = DTR0.mean(1)
    muTR1 = DTR1.mean(1)

    CTR0 = covarianceMatrix(DTR0)
    CTR1 = covarianceMatrix(DTR1)

    tot = LTR.shape[0]
    num1 = np.sum(LTR)
    num0 = tot - num1

    CTC = (num0 * CTR0 + num1 * CTR1) / tot

    logpdf0_TC = logpdf_GAU_ND(DTE, vcol(muTR0), CTC)
    logpdf1_TC = logpdf_GAU_ND(DTE, vcol(muTR1), CTC)

    return np.vstack([logpdf0_TC,logpdf1_TC])

def TCGC_KFold(D, L, k=3):
    folds, labels = Ksplit(D, L, seed=0, K=k)
    orderedLabels = []
    scores = []
    for i in range(k):
        trainingSet = []
        labelsOfTrainingSet = []
        for j in range(k):
            if j != i:
                trainingSet.append(folds[j])
                labelsOfTrainingSet.append(labels[j])
        evaluationSet = folds[i]
        orderedLabels.append(labels[i])
        trainingSet = np.hstack(trainingSet)
        labelsOfTrainingSet = np.hstack(labelsOfTrainingSet)
        scores.append(predictllr(trainingSet, labelsOfTrainingSet, evaluationSet))  #tiedCovarianceGC
    scores = np.hstack(scores)
    orderedLabels = np.hstack(orderedLabels)
    labels = np.hstack(labels)
    return (scores, orderedLabels)

def confMat(predLabels,labels,nClasses):
    confMat = np.zeros((nClasses, nClasses))
    #preCor = [(predLabels[i], LTE[i]) for i in range(predLabels.shape[0])]
    for i in range(nClasses):
        predH = predLabels == i
        for j in range(nClasses):
            corrH = labels == j
            confMat[i, j] = np.dot(predH.astype(int), corrH.astype(int).T)
    return confMat

def predictLabels(p1,Cfn,Cfp,llr):
    t = -np.log(p1 * Cfn / ((1 - p1) * Cfp))
    predLabels = llr > t

    return predLabels
def computeBayesRisk(p1,Cfn,Cfp,M):
    FPR=M[1,0]/(M[0,0]+M[1,0])
    FNR=M[0,1]/(M[0,1]+M[1,1])
    DCF=p1*Cfn*FNR+(1-p1)*Cfp*FPR

    return DCF

def computeMinDCF(p1,Cfn,Cfp,llr,labels):
    nClasses=max(labels)+1
    Bnorm = min(p1 * Cfn, (1 - p1) * Cfp)
    pred_t = [llr > t for t in sorted(llr)]
    M_t = [confMat(pred_t[i], labels, nClasses) for i in range(len(llr))]
    DCF_t = [computeBayesRisk(p1, Cfn, Cfp, M_t[i]) for i in range(len(llr))]
    normDCF_t = np.array(DCF_t) / Bnorm

    return min(normDCF_t)
def computeNormDCF(p1,Cfn,Cfp,llr,labels):
    Bnorm = min(p1 * Cfn, (1 - p1) * Cfp)

    predLabels = np.array(predictLabels(p1, Cfn, Cfp, llr)).astype(int)
    confM1 = confMat(predLabels, labels, 2)
    DCF = computeBayesRisk(p1, Cfn, Cfp, confM1)
    normDCF = DCF / Bnorm

    return confM1,DCF,normDCF

#LOGISTIC REGRESSION
def logreg_obj(v,DTR,LTR,l,pi_t=0.5):

    w,b=v[0:-1],v[-1]
    vsum=0
    vsum1=0
    vsum0=0
    nt=np.sum(LTR == 1)
    nf=np.sum(LTR == 0)
    for x_i in DTR[:, LTR == 1].T:
        vsum1+= np.logaddexp(0, -1 * (np.dot(w, x_i) + b))
    for x_i in DTR[:, LTR == 0].T:
        vsum0 += np.logaddexp(0, 1 * (np.dot(w, x_i) + b))

    for i in range(len(LTR)):
        vsum+=np.logaddexp(0,-(2*LTR[i]-1)*(np.dot(w,DTR[:,i])+b))
    return l/2. * np.dot(w.T,w)+pi_t/nt*vsum1+(1-pi_t)/nf*vsum0

def binaryLogisticRegression(DTR, LTR, DTE, lamb,pi_t):
    x0 = np.zeros(DTR.shape[0] + 1)
    #train
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, lamb,pi_t), approx_grad=True)

    w, b = x[0:-1], x[-1]

    return np.dot(w.T,DTE)+b #evaluate

#GMM
def GMMClf(DTR, LTR, M, EMStop=1e-6, alpha=0.1, psi=0.01, diag=0, tied=0):
    splitnum = int(np.log2(M))
    # GMMdata0=DTR[:,LTR==0]
    # GMMdata1=DTR[:,LTR==1]
    # mu0,S0=compute_mu_s(GMMdata0)
    # mu1, S1= compute_mu_s(GMMdata1)

    k = 3
    folds, labels = Ksplit(DTR, LTR, seed=0, K=3)
    orderedLabels = []
    scores =[] # [[] for i in range(splitnum)]
    # if PCA_flag:
    for i in range(k):
        # trainingSet=[folds[j] for j in range(k) if j!=i]
        trainingSet = []
        labelsOfTrainingSet = []
        for j in range(k):
            if j != i:
                trainingSet.append(folds[j])
                labelsOfTrainingSet.append(labels[j])
        evaluationSet = folds[i]
        orderedLabels.append(labels[i])
        trainingSet = np.hstack(trainingSet)
        labelsOfTrainingSet = np.hstack(labelsOfTrainingSet)
        trainSet0 = trainingSet[:, labelsOfTrainingSet == 0]
        trainSet1 = trainingSet[:, labelsOfTrainingSet == 1]
        mu0, S0 = compute_mu_s(trainSet0)
        mu1, S1 = compute_mu_s(trainSet1)
        GMMList0 = LBGAlgorithm(trainSet0, mu0, constrainCovMat([S0], 1, psi)[0], EMStop, alpha, splitnum, diag, tied)
        GMMList1 = LBGAlgorithm(trainSet1, mu1, constrainCovMat([S1], 1, psi)[0], EMStop, alpha, splitnum, diag, tied)

        #for splitIdx in range(splitnum):
            # crea vettore riga degli score di ogni split
        scores.append(computellr(evaluationSet, GMMList0[-1], GMMList1[-1]))
    orderedLabels = np.hstack(orderedLabels)
    scores=np.hstack(scores)
    return scores,orderedLabels

def calibrateScoresLogReg(scores,orderedLabels,l,prior=0.5):
    # np.log(prior/(1-prior))
    scores=vrow(scores)
    calScores = binaryLogisticRegression(scores, orderedLabels, scores, l, prior) - np.log(prior / (1 - prior))

    return calScores

def printBayesErrorPlot(scores,orderedLabels,legend):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    piVet = 1 / (1 + np.exp(-effPriorLogOdds))

    normDCF_p = [computeNormDCF(piVet[i], 1, 1, scores, orderedLabels)[2] for i in range(len(piVet))]
    minDCF_p = [computeMinDCF(piVet[i], 1, 1, scores, orderedLabels) for i in range(len(piVet))]

    plt.figure()
    plt.plot(effPriorLogOdds, normDCF_p, label=legend[0], color='r')  # no pseudocounts
    plt.plot(effPriorLogOdds, minDCF_p, label=legend[1], color='b')
    # plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.xlabel('prior log odds')
    plt.ylabel('DCF value')
    plt.show()

if __name__=="__main__":
    PCA_flag = False
    PCA_factor = 7
    TCGC_flag = False
    LogRegFlag=False
    GMMFLag=True
    BayesErrPlotFlag=True
    calibratedPlotFlag=False

    priors = [0.5, 0.9, 0.1]

    # load of training and test set with labels
    (DTR, LTR), (DTE, LTE) = loadPulsar()

    # Z normalization of data
    DTR, meanDTR, standardDeviationDTR = ZNormalization(DTR)
    DTE, meanDTE, standardDeviationDTE = ZNormalization(DTE,meanDTR, standardDeviationDTR)

    # principal component analisys
    if PCA_flag:
        DTR = PCA(DTR, PCA_factor)
        DTE = PCA(DTE, PCA_factor)

    CalRes = np.empty((3, 6))
    calibratedRes=np.empty((3,6))
    if TCGC_flag:
        (scores,orderedLabels)=TCGC_KFold(DTR,LTR)
        predLabels = np.argmax(np.exp(scores), 0)
        print('TIED COV MVG')
        for i,prior in enumerate(priors):
            #DCF=computeDCF(Conf,1,1,prior)
            llr = scores[1, :] - scores[0, :]

            calScoresTCGC = calibrateScoresLogReg(llr, orderedLabels, 1e-4, 0.5)

            minDCF = computeMinDCF(prior, 1, 1, llr, orderedLabels)
            _,_,actDCF= computeNormDCF(prior,1,1,llr,orderedLabels)

            CalminDCF = computeMinDCF(prior, 1, 1, calScoresTCGC, orderedLabels)
            _, _, CalactDCF = computeNormDCF(prior, 1, 1, calScoresTCGC, orderedLabels)

            #Bdummy=computeBDummy(1,1,prior)
            print("For prior probability: pi=",prior)
            print("Normalized min DCF:",minDCF)
            print('Actual norm dcf: ',actDCF)

            print('calibrated min DCF: ',CalminDCF)
            print('calibrated actual dcf: ',CalactDCF)

            CalRes[0, 2*i] = minDCF
            CalRes[0, 2*i+1]=actDCF

            calibratedRes[0, 2*i]=CalminDCF
            calibratedRes[0,2*i+1]=CalactDCF
        #BAYES ERROR PLOT
        if BayesErrPlotFlag:
            printBayesErrorPlot(llr,orderedLabels,('Tied MVG actual DCF','Tied MVG min DCF'))
        if calibratedPlotFlag:
            printBayesErrorPlot(calScoresTCGC, orderedLabels, ('Tied MVG calibrated actual DCF', 'Tied MVG min DCF'))

    if LogRegFlag:
        k=3
        l=1e-4  #lambda hyperparameter
        folds, labels = Ksplit(DTR, LTR, seed=0, K=k)
        orderedLabels=[]
        scores5=[]
        for i in range(k):
            # trainingSet=[folds[j] for j in range(k) if j!=i]
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(k):
                if j != i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet = np.hstack(trainingSet)
            labelsOfTrainingSet = np.hstack(labelsOfTrainingSet)
            #scores1.append(binaryLogisticRegression(trainingSet, labelsOfTrainingSet, evaluationSet, l, 0.1))
            scores5.append(binaryLogisticRegression(trainingSet, labelsOfTrainingSet, evaluationSet, l, 0.5))
            #scores9.append(binaryLogisticRegression(trainingSet, labelsOfTrainingSet, evaluationSet, l, 0.9))
        scores5 = np.hstack(scores5)
        orderedLabels = np.hstack(orderedLabels)

        calScoresLR=calibrateScoresLogReg(scores5,orderedLabels,l,0.5)

        dcf51 = computeMinDCF(0.1, 1, 1, scores5, orderedLabels)
        dcf55 = computeMinDCF(0.5, 1, 1, scores5, orderedLabels)
        dcf59 = computeMinDCF(0.9, 1, 1, scores5, orderedLabels)
        print('Logistic Regression - prior: 0.5 - 0.9 - 0.1')
        print('lambda=1e-4 pi_t=0.5 min DCF: ', dcf55, dcf59, dcf51)
        _, _, actDCF5 = computeNormDCF(0.5, 1, 1, scores5, orderedLabels)
        _, _, actDCF9 = computeNormDCF(0.9, 1, 1, scores5, orderedLabels)
        _, _, actDCF1 = computeNormDCF(0.1, 1, 1, scores5, orderedLabels)
        CalRes[1] = [dcf55, actDCF5,dcf59,actDCF9,dcf51,actDCF1]
        print('lambda=1e-4 pi_t=0.5 actual DCF: ',actDCF5,actDCF9,actDCF1)

        for i,prior in enumerate(priors):
            CalminDCF = computeMinDCF(prior, 1, 1, calScoresLR, orderedLabels)
            _, _, CalactDCF = computeNormDCF(prior, 1, 1, calScoresLR, orderedLabels)

            print('calibrated minDCF: ', CalminDCF)
            print('calibrated actual DCF: ', CalactDCF)

            calibratedRes[1, 2 * i] = CalminDCF
            calibratedRes[1, 2 * i + 1] = CalactDCF

        #BAYES ERROR PLOT
        if BayesErrPlotFlag:
            printBayesErrorPlot(scores5,orderedLabels,('Logistic Regression actual DCF','Logistic Regression min DCF'))
        if calibratedPlotFlag:
            printBayesErrorPlot(calScoresLR, orderedLabels, ('Logistic Regression calibrated actual DCF', 'Logistic Regression min DCF'))

    if GMMFLag:
        EMStop = 1e-6
        M = 8  # numero massimo di componenti

        scores,orderedLabels = GMMClf(DTR, LTR, M)

        calScoresGMM = calibrateScoresLogReg(scores, orderedLabels, 1e-4, 0.5)
        #dcf1 = []
        #dcf5 = []
        #dcf9 = []
        dcf1=computeMinDCF(0.1, 1, 1, scores, orderedLabels)
        dcf5=computeMinDCF(0.5, 1, 1, scores, orderedLabels)
        dcf9=computeMinDCF(0.9, 1, 1, scores, orderedLabels)

        print('GMM 8 components model:')
        print('min dcf 0.1-0.5-0.9: ',dcf1,dcf5,dcf9)
        #print('min dcf 0.5:', dcf5)
        #print('min dcf 0.9:', dcf9)
        _, _, actDCF5 = computeNormDCF(0.5, 1, 1, scores, orderedLabels)
        _, _, actDCF9 = computeNormDCF(0.9, 1, 1, scores, orderedLabels)
        _, _, actDCF1 = computeNormDCF(0.1, 1, 1, scores, orderedLabels)
        CalRes[2] = [dcf5, actDCF5, dcf9, actDCF9, dcf1, actDCF1]
        print('actual dcf 0.1-0.5-0,9: ',actDCF1,actDCF5,actDCF9)

        for i, prior in enumerate(priors):
            CalminDCF = computeMinDCF(prior, 1, 1, calScoresGMM, orderedLabels)
            _, _, CalactDCF = computeNormDCF(prior, 1, 1, calScoresGMM, orderedLabels)

            print('calibrated minDCF (prior %f): '%prior, CalminDCF)
            print('calibrated actual DCF (prior %f): '%prior, CalactDCF)

            calibratedRes[2, 2 * i] = CalminDCF
            calibratedRes[2, 2 * i + 1] = CalactDCF
        if BayesErrPlotFlag:
            printBayesErrorPlot(scores,orderedLabels,('GMM 8 components actual DCF','GMM 8 components min DCF'))
        if calibratedPlotFlag:
            printBayesErrorPlot(calScoresGMM, orderedLabels, ('GMM 8 components actual calibrated DCF', 'GMM 8 components min DCF'))



    print('                  prior:     0.5     -     0.9     -     0.1')
    print('                        minDCF actDCF minDCF actDCF minDCF actDCF')
    print('MVG tied-cov              %.3f %.3f %.3f %.3f %.3f %.3f'% (CalRes[0][0],CalRes[0][1],CalRes[0][2],CalRes[0][3],CalRes[0][4],CalRes[0][5]))
    print('LogReg l=1e-4, pi_t=0.5   %.3f %.3f %.3f %.3f %.3f %.3f'% (CalRes[1][0],CalRes[1][1],CalRes[1][2],CalRes[1][3],CalRes[1][4],CalRes[1][5]))
    print('GMM full-cov 8 components %.3f %.3f %.3f %.3f %.3f %.3f'% (CalRes[2][0],CalRes[2][1],CalRes[2][2],CalRes[2][3],CalRes[2][4],CalRes[2][5]))

    print('RESULTS AFTER CALIBRATION')
    print('                  prior:     0.5     -     0.9     -     0.1')
    print('                        minDCF actDCF minDCF actDCF minDCF actDCF')
    print('MVG tied-cov              %.3f %.3f %.3f %.3f %.3f %.3f' % (
    calibratedRes[0][0], calibratedRes[0][1], calibratedRes[0][2], calibratedRes[0][3], calibratedRes[0][4], calibratedRes[0][5]))
    print('LogReg l=1e-4, pi_t=0.5   %.3f %.3f %.3f %.3f %.3f %.3f' % (
    calibratedRes[1][0], calibratedRes[1][1], calibratedRes[1][2], calibratedRes[1][3], calibratedRes[1][4], calibratedRes[1][5]))
    print('GMM full-cov 8 components %.3f %.3f %.3f %.3f %.3f %.3f' % (
    calibratedRes[2][0], calibratedRes[2][1], calibratedRes[2][2], calibratedRes[2][3], calibratedRes[2][4], calibratedRes[2][5]))

