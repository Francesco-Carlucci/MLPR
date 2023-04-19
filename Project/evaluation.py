import numpy
import matplotlib.pyplot as plt
from util import loadPulsar,PCA,ZNormalization,GC,naiveBayesGC,tiedCovarianceGC
from calibration import predictllr ,calibrateScoresLogReg,computeMinDCF,computeNormDCF
from LogReg import  binaryLogisticRegression
from GMM import computellr,compute_mu_s,LBGAlgorithm,constrainCovMat,confMat

def ROCPlot(llr,labels):
    nClasses=max(labels)+1
    pred_t = [llr > t for t in sorted(llr)]
    M_t = [confMat(pred_t[i], labels, nClasses) for i in range(len(llr))]
    FPR_t = [M_t[i][1, 0] / (M_t[i][1, 0] + M_t[i][0, 0]) for i in range(len(llr))]
    TPR_t = [1 - (M_t[i][0, 1] / (M_t[i][0, 1] + M_t[i][1, 1])) for i in range(len(llr))]

    return FPR_t,TPR_t

if __name__=="__main__":
    PCA_flag = True
    PCA_factor = 7
    GC_flag = False
    NBGC_flag=False
    TCGC_flag=True
    LR_flag=True
    GMM_flag=True
    DiagGMM_flag=False
    TiedGMM_flag=False

    priors = [0.5, 0.9, 0.1]

    # load of training and test set with labels
    (DTR, LTR), (DTE, LTE) = loadPulsar()

    # Z normalization of data
    DTR, meanDTR, standardDeviationDTR = ZNormalization(DTR)
    DTE, meanDTE, standardDeviationDTE = ZNormalization(DTE, meanDTR, standardDeviationDTR)

    fig = plt.figure()

    # principal component analisys
    DTRnoPCA=DTR
    DTEnoPCA=DTE
    if PCA_flag:
        DTR = PCA(DTR, PCA_factor)
        DTE = PCA(DTE, PCA_factor)

    if GC_flag:
        scores=GC(DTR, LTR, DTE)
        llr = scores[1, :] - scores[0, :]
        print('Full Cov MVG: ', [computeMinDCF(prior, 1, 1, llr, LTE) for prior in priors])

    if NBGC_flag:
        scores = naiveBayesGC(DTR, LTR, DTE)
        llr = scores[1, :] - scores[0, :]
        print('Diag Cov MVG: ', [computeMinDCF(prior, 1, 1, llr, LTE) for prior in priors])

    if TCGC_flag:
        scores=tiedCovarianceGC(DTR, LTR, DTE)
        llr = scores[1, :] - scores[0, :]
        print('Tied Cov MVG: ', [computeMinDCF(prior, 1, 1, llr, LTE) for prior in priors])

        FPR_t,TPR_t=ROCPlot(llr,LTE)

        plt.plot(FPR_t, TPR_t, label='Tied Cov MVG')
        plt.xlabel('FPR')
        plt.ylabel('TPR')

    if LR_flag:
        scores=binaryLogisticRegression(DTR,LTR,DTE,1e-4,0.5)
        print('Logistic Regression: ', [computeMinDCF(prior, 1, 1, scores, LTE) for prior in priors])

        FPR_t,TPR_t=ROCPlot(scores,LTE)
        plt.plot(FPR_t, TPR_t, label='Logistic regression ')

    if GMM_flag:
        EMStop = 1e-6
        Mmax = 16
        splitnum=4
        psi=0.01
        alpha=0.1

        trainSet0=DTR[:, LTR == 0]
        trainSet1=DTR[:,LTR==1]
        mu0, S0 = compute_mu_s(trainSet0)
        mu1, S1 = compute_mu_s(trainSet1)

        GMMList0 = LBGAlgorithm(trainSet0, mu0, constrainCovMat([S0], 1, psi)[0], EMStop, alpha, splitnum, 0, 0)
        GMMList1 = LBGAlgorithm(trainSet1, mu1, constrainCovMat([S1], 1, psi)[0], EMStop, alpha, splitnum, 0, 0)
        #scores=[]
        for splitIdx in range(splitnum):
            scores=computellr(DTE, GMMList0[splitIdx], GMMList1[splitIdx])
            print('GMM Full %d components: '%(2**(splitIdx+1)), [computeMinDCF(prior, 1, 1, scores, LTE) for prior in priors])
            """
            if splitIdx==3:
                scores = computellr(DTEnoPCA, GMMList0[splitIdx], GMMList1[splitIdx])
                FPR_t,TPR_t=ROCPlot(scores,LTE)
                plt.plot(FPR_t, TPR_t, label='GMM 8 components')
            """

    if DiagGMM_flag:
        EMStop = 1e-6
        Mmax = 16
        splitnum=4
        psi=0.01
        alpha=0.1

        trainSet0=DTR[:, LTR == 0]
        trainSet1=DTR[:,LTR==1]
        mu0, S0 = compute_mu_s(trainSet0)
        mu1, S1 = compute_mu_s(trainSet1)

        GMMList0 = LBGAlgorithm(trainSet0, mu0, constrainCovMat([S0], 1, psi)[0], EMStop, alpha, splitnum, 1, 0)
        GMMList1 = LBGAlgorithm(trainSet1, mu1, constrainCovMat([S1], 1, psi)[0], EMStop, alpha, splitnum, 1, 0)
        #scores=[]
        for splitIdx in range(splitnum):
            scores=computellr(DTE, GMMList0[splitIdx], GMMList1[splitIdx])
            print('GMM Diag %d components: '%(2**(splitIdx+1)), [computeMinDCF(prior, 1, 1, scores, LTE) for prior in priors])
    if TiedGMM_flag:
        EMStop = 1e-6
        Mmax = 16
        splitnum = 4
        psi = 0.01
        alpha = 0.1

        trainSet0 = DTR[:, LTR == 0]
        trainSet1 = DTR[:, LTR == 1]
        mu0, S0 = compute_mu_s(trainSet0)
        mu1, S1 = compute_mu_s(trainSet1)

        GMMList0 = LBGAlgorithm(trainSet0, mu0, constrainCovMat([S0], 1, psi)[0], EMStop, alpha, splitnum, 0, 1)
        GMMList1 = LBGAlgorithm(trainSet1, mu1, constrainCovMat([S1], 1, psi)[0], EMStop, alpha, splitnum, 0, 1)
        # scores=[]
        for splitIdx in range(splitnum):
            scores = computellr(DTE, GMMList0[splitIdx], GMMList1[splitIdx])
            print('GMM Diag %d components: ' % (2 ** (splitIdx + 1)),
                  [computeMinDCF(prior, 1, 1, scores, LTE) for prior in priors])

    EMStop = 1e-6
    Mmax = 8
    splitnum = 4
    psi = 0.01
    alpha = 0.1

    trainSet0 = DTRnoPCA[:, LTR == 0]
    trainSet1 = DTRnoPCA[:, LTR == 1]
    mu0, S0 = compute_mu_s(trainSet0)
    mu1, S1 = compute_mu_s(trainSet1)

    GMMList0 = LBGAlgorithm(trainSet0, mu0, constrainCovMat([S0], 1, psi)[0], EMStop, alpha, splitnum, 0, 0)
    GMMList1 = LBGAlgorithm(trainSet1, mu1, constrainCovMat([S1], 1, psi)[0], EMStop, alpha, splitnum, 0, 0)
    # scores=[]
    scores = computellr(DTEnoPCA, GMMList0[-1], GMMList1[-1])
    FPR_t, TPR_t = ROCPlot(scores, LTE)
    plt.plot(FPR_t, TPR_t, label='GMM 8 components')
    plt.legend()
    plt.show()

