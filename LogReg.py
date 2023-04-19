import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from util import loadPulsar,PCA,ZNormalization,Ksplit
#per parallelizzare i grafici lunghi
import os
import time
from concurrent.futures import ProcessPoolExecutor,as_completed

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

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

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, lamb,pi_t), approx_grad=True)

    w, b = x[0:-1], x[-1]

    S = []
    """
    for i in range(DTE.shape[1]):
        S.append(numpy.dot(w.T, DTE[:, i]) + b)
    """
    return np.dot(w.T,DTE)+b

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

def logregLambdaplot(lambdaList,DTR,LTR,index):
    dcf1 = []
    dcf5 = []
    dcf9 = []
    #lambdaList = np.linspace(1e-5, 1e2, 1000)
    k = 3
    for l in lambdaList:
        folds, labels = Ksplit(DTR, LTR, seed=0, K=k)
        orderedLabels = []
        scores = []
        #if PCA_flag:
        for i in range(k):
            #trainingSet=[folds[j] for j in range(k) if j!=i]
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
            scores.append(binaryLogisticRegression(trainingSet, labelsOfTrainingSet, evaluationSet, l,0.5))
        scores = np.hstack(scores)
        orderedLabels = np.hstack(orderedLabels)
        """
        else:
            scores = binaryLogisticRegression(DTR, LTR, DTE, l,0.5)
            orderedLabels=LTE
        """
        dcf1.append(computeMinDCF(0.1, 1, 1, scores, orderedLabels))
        dcf5.append(computeMinDCF(0.5, 1, 1, scores, orderedLabels))
        dcf9.append(computeMinDCF(0.9, 1, 1, scores, orderedLabels))

    return (np.vstack([dcf1,dcf5,dcf9]),index)

def scheduleTask(mainTask,workingVet,DTR,LTR):
    resdict={}
    nWorkers= os.cpu_count()
    print('number of workers: ',nWorkers)
    start=time.time()
    with ProcessPoolExecutor(max_workers=nWorkers) as executor:

        elementsPerThread=len(workingVet)//nWorkers

        futures=[executor.submit(mainTask,workingVet[i*elementsPerThread:(i+1)*elementsPerThread],DTR,LTR,i) for i in range(nWorkers)]
        if nWorkers*elementsPerThread<len(workingVet):
            futures.append(executor.submit(mainTask,workingVet[nWorkers*elementsPerThread:len(workingVet)]))

        for future in as_completed(futures):
            data=future.result()
            data,index=data
            resdict[index]=data
            print(data)

    #wait(futures)
    resVet=[resdict[k] for k in sorted(resdict.keys())]
    resVet=np.hstack(resVet)
    print('durata: ',time.time()-start)
    return resVet

if __name__=='__main__':
    PCA_flag = False
    PCA_factor = 7
    logRegPlot = False

    priors = [0.5, 0.9, 0.1]

    # load of training and test set with labels
    (DTR, LTR), (DTE, LTE) = loadPulsar()

    # Z normalization of data
    DTR, meanDTR, standardDeviationDTR = ZNormalization(DTR)
    DTE, meanDTE, standardDeviationDTE = ZNormalization(DTE)

    if PCA_flag:
        DTR=PCA(DTR,PCA_factor)
        DTE=PCA(DTE,PCA_factor)

    if logRegPlot:
        #PLOT LOGISTIC REGRESSION LAMBDA
        lambdaList=np.logspace(np.log10(1e-5),np.log10(1e2),100)
        print('lambdalist',lambdaList)
        results=scheduleTask(logregLambdaplot,lambdaList,DTR,LTR)
        #results = logregLambdaplot(lambdaList, DTR, LTR, DTE, LTE)
        dcf1=results[0]
        dcf5=results[1]
        dcf9=results[2]
        plt.figure()
        print('dcf1:',dcf1)
        print('dcf5:', dcf5)
        print('dcf9:', dcf9)
        plt.plot(lambdaList,dcf1,label='minDCF prior=0.1')
        plt.plot(lambdaList,dcf5,label='minDCF prior=0.5')
        plt.plot(lambdaList,dcf9,label='minDCF prior=0.9')
        plt.legend()
        plt.xlabel('λ')
        plt.ylabel('min DCF')
        plt.xscale('log')
        plt.show() # λ

    #TABELLA LOGISTIC REGRESSION PRIOR PI_T
    k=3
    l=1e-4
    folds, labels = Ksplit(DTR, LTR, seed=0, K=k)
    orderedLabels = []
    scores1 = []
    scores5 = []
    scores9 = []
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
        scores1.append(binaryLogisticRegression(trainingSet, labelsOfTrainingSet, evaluationSet, l,0.1))
        scores5.append(binaryLogisticRegression(trainingSet, labelsOfTrainingSet, evaluationSet, l, 0.5))
        scores9.append(binaryLogisticRegression(trainingSet, labelsOfTrainingSet, evaluationSet, l, 0.9))

    scores1=np.hstack(scores1)
    scores5=np.hstack(scores5)
    scores9=np.hstack(scores9)
    orderedLabels = np.hstack(orderedLabels)
    if PCA_flag:print('LogReg PCA: ',PCA_factor)
    else:print('LogReg no PCA')

    dcf11=computeMinDCF(0.1, 1, 1, scores1, orderedLabels)
    dcf15=computeMinDCF(0.5, 1, 1, scores1, orderedLabels)
    dcf19=computeMinDCF(0.9, 1, 1, scores1, orderedLabels)
    print('lambda=1e-4 pi_t=0.1',dcf15,dcf19,dcf11)
    dcf51 = computeMinDCF(0.1, 1, 1, scores5, orderedLabels)
    dcf55 = computeMinDCF(0.5, 1, 1, scores5, orderedLabels)
    dcf59 = computeMinDCF(0.9, 1, 1, scores5, orderedLabels)
    print('lambda=1e-4 pi_t=0.5', dcf55, dcf59, dcf51)
    dcf91 = computeMinDCF(0.1, 1, 1, scores9, orderedLabels)
    dcf95 = computeMinDCF(0.5, 1, 1, scores9, orderedLabels)
    dcf99 = computeMinDCF(0.9, 1, 1, scores9, orderedLabels)
    print('lambda=1e-4 pi_t=0.9', dcf95, dcf99, dcf91)