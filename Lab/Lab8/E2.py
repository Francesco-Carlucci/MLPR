import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

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
    Bnorm = min(p1 * Cfn, (1 - p1) * Cfp)
    pred_t = [llr > t for t in sorted(llr)]
    M_t = [confMat(pred_t[i], labels, nClasses) for i in range(len(llr))]
    DCF_t = [computeBayesRisk(p1, Cfn, Cfp, M_t[i]) for i in range(len(llr))]
    normDCF_t = np.array(DCF_t) / Bnorm

    return min(normDCF_t)
def computeNormDCF(p1,Cfn,Cfp,llr,labels):
    Bnorm = min(p1 * Cfn, (1 - p1) * Cfp)

    predLabels = predictLabels(p1, Cfn, Cfp, llr)
    confM1 = confMat(predLabels, labels, nClasses)
    DCF = computeBayesRisk(p1, Cfn, Cfp, confM1)
    normDCF = DCF / Bnorm

    return confM1,DCF,normDCF

def computeDCFmulticlass(ll_multiclass,labels_multiclass,pi,C,):
    normCost = np.min(np.dot(C, pi))

    SJoint = ll_multiclass + np.log(pi)
    SMarginal = special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SMarginal
    costVect = np.dot(C, np.exp(SPost))
    predLabels = np.argmin(costVect, axis=0)

    M = confMat(predLabels, labels_multiclass, 3)

    R = M / np.sum(M, axis=0)

    DCF = np.dot(pi.T, np.sum(R * C, axis=0))

    return M,DCF,DCF/normCost

if __name__=='__main__':
    llr_infpar=np.load('Data/commedia_llr_infpar.npy')
    labels_infpar=np.load('Data/commedia_labels_infpar.npy')

    nClasses=2
    p1=0.5
    Cfn=1
    Cfp=1
    Bnorm=min(p1*Cfn,(1-p1)*Cfp)

    confM1,DCF,normDCF1=computeNormDCF(0.5,1,1,llr_infpar,labels_infpar)
    print('Confusion matrix pi1: %.2f Cfn: %d Cfp: %d :\n'%(p1,Cfn,Cfp),confM1)
    print('Bayes risk: ', DCF,' normalized: ',normDCF1)

    confM2, DCF, normDCF2 = computeNormDCF(0.8, 1, 1, llr_infpar, labels_infpar)
    print('Confusion matrix pi1: %.2f Cfn: %d Cfp: %d :\n' % (p1, Cfn, Cfp), confM2)
    print('Bayes risk: ', DCF,' normalized: ',normDCF2)

    confM3, DCF, normDCF3 = computeNormDCF(0.5, 10, 1, llr_infpar, labels_infpar)
    print('Confusion matrix pi1: %.2f Cfn: %d Cfp: %d :\n' % (p1, Cfn, Cfp), confM3)
    print('Bayes risk: ',DCF,' normalized: ',normDCF3)

    p1=0.8
    Cfn=1
    Cfp=10
    confM4, DCF, normDCF4 = computeNormDCF(0.8, 1, 10, llr_infpar, labels_infpar)
    print('Confusion matrix pi1: %.2f Cfn: %d Cfp: %d :\n' % (p1, Cfn, Cfp), confM4)
    print('Bayes risk: ', DCF,' normalized: ',normDCF4)

    minDCF1=computeMinDCF(0.5,1,1,llr_infpar,labels_infpar)
    print('min DCF: ', minDCF1)
    minDCF2 = computeMinDCF(0.8, 1, 1, llr_infpar, labels_infpar)
    print('min DCF: ', minDCF2)
    minDCF3 = computeMinDCF(0.5, 10, 1, llr_infpar, labels_infpar)
    print('min DCF: ', minDCF3)
    minDCF4 = computeMinDCF(0.8, 1, 10, llr_infpar, labels_infpar)
    print('min DCF: ', minDCF4)

    #ROC plot
    pred_t = [llr_infpar > t for t in sorted(llr_infpar)]
    M_t = [confMat(pred_t[i], labels_infpar, nClasses) for i in range(len(llr_infpar))]
    FPR_t=[M_t[i][1,0]/(M_t[i][1,0]+M_t[i][0,0]) for i in range(len(llr_infpar))]
    TPR_t=[1-(M_t[i][0,1]/(M_t[i][0,1]+M_t[i][1,1])) for i in range(len(llr_infpar))]

    fig=plt.figure()
    plt.plot(FPR_t,TPR_t)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

    effPriorLogOdds = np.linspace(-3, 3,21)
    piVet=1/(1+np.exp(-effPriorLogOdds))

    normDCF_p= [computeNormDCF(piVet[i], 1, 1, llr_infpar, labels_infpar)[2] for i in range(len(piVet))]
    """
    Bnorm = [min(piVet[i], (1 - piVet[i])) for i in range(len(piVet))]
    pred_p = [predictLabels(piVet[i], 1, 1, llr_infpar) for i in range(len(piVet))]
    confM_p = [confMat(pred_p[i], labels_infpar, nClasses) for i in range(len(piVet))]
    DCF_p = [computeBayesRisk(piVet[i], 1, 1, confM_p[i]) for i in range(len(piVet))]
    normDCF_p = [DCF_p[i] / Bnorm[i] for i in range(len(piVet))]
    """
    minDCF_p=[computeMinDCF(piVet[i],1,1,llr_infpar,labels_infpar) for i in range(len(piVet))]

    plt.figure()
    plt.plot(effPriorLogOdds,normDCF_p,label='DCF (ε=0.001)',color='r')
    plt.plot(effPriorLogOdds,minDCF_p,label='min DCF (ε=0.001)',color='b')
    plt.ylim([0,1.1])
    plt.xlim([-3,3])
    plt.xlabel('prior log odds')
    plt.ylabel('DCF value')

    llr_infpar_eps1=np.load('Data/commedia_llr_infpar_eps1.npy')

    normDCF_p_eps1 = [computeNormDCF(piVet[i], 1, 1, llr_infpar_eps1, labels_infpar)[2] for i in range(len(piVet))]
    minDCF_p_eps1 = [computeMinDCF(piVet[i], 1, 1, llr_infpar_eps1, labels_infpar) for i in range(len(piVet))]

    plt.plot(effPriorLogOdds,normDCF_p_eps1,label='DCF (ε=1)',color='y')
    plt.plot(effPriorLogOdds,minDCF_p_eps1,label='min DCF (ε=1)',color='g')
    plt.legend()
    plt.show()

    #MULTICLASS

    ll_multiclass=np.load('Data/commedia_ll.npy')
    labels_multiclass=np.load('Data/commedia_labels.npy')

    C=np.array([[0,1,2],[1,0,1],[2,1,0]])
    pi=vcol(np.array([0.3,0.4,0.3]))

    M,DCF,normDCF=computeDCFmulticlass(ll_multiclass,labels_multiclass,pi,C)

    print('M=\n',M,'DCF_u multiclass: ',DCF,' DCF: ',normDCF)

    ll_multiclass_eps1=np.load('Data/commedia_ll_eps1.npy')

    M_1,DCF_1,normDCF_1=computeDCFmulticlass(ll_multiclass_eps1,labels_multiclass,pi,C)

    print('M=\n',M_1,'DCF_u multiclass (eps=1.0): ', DCF_1, ' DCF (eps=1.0): ', normDCF_1)

    pi=np.array([[1./3.],[1./3.],[1./3.]])
    C=1-np.eye(3)
    M, DCF, normDCF = computeDCFmulticlass(ll_multiclass, labels_multiclass, pi, C)
    M_1, DCF_1, normDCF_1 = computeDCFmulticlass(ll_multiclass_eps1, labels_multiclass, pi, C)

    print('M=\n', M, 'DCF_u multiclass (eps=1.0): ', DCF, ' DCF (eps=1.0): ', normDCF)
    print('M=\n', M_1, 'DCF_u multiclass (eps=1.0): ', DCF_1, ' DCF (eps=1.0): ', normDCF_1)
