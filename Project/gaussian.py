"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""
import numpy
import string
import scipy.special
import itertools
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
from itertools import repeat
import scipy.special as scsp
from util import loadPulsar,PCA,LDA,GC,calculatePrecision,calculateConfusionMatrix,naiveBayesGC,tiedCovarianceGC,tiedCovarianceNaiveBayesGC,ZNormalization,computeDCF,computeBDummy,Ksplit

from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import os
import time

def GC_KFold(D,L,k=3):
    
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
         scores.append(GC(trainingSet,labelsOfTrainingSet,evaluationSet))
    scores=np.hstack(scores)
    orderedLabels=np.hstack(orderedLabels)
    labels = np.hstack(labels)
    return (scores, orderedLabels)

def NBGC_KFold(D,L,k=3):
    
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
         scores.append(naiveBayesGC(trainingSet,labelsOfTrainingSet,evaluationSet))
    scores=np.hstack(scores)
    orderedLabels=np.hstack(orderedLabels)
    labels = np.hstack(labels)
    return (scores, orderedLabels)

def TCGC_KFold(D,L,k=3):
    
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
         scores.append(tiedCovarianceGC(trainingSet,labelsOfTrainingSet,evaluationSet))
    scores=np.hstack(scores)
    orderedLabels=np.hstack(orderedLabels)
    labels = np.hstack(labels)
    return (scores, orderedLabels)

def TCNBGC_KFold(D,L,k=3):
    
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
         scores.append(tiedCovarianceNaiveBayesGC(trainingSet,labelsOfTrainingSet,evaluationSet))
    scores=np.hstack(scores)
    orderedLabels=np.hstack(orderedLabels)
    labels = np.hstack(labels)
    return (scores, orderedLabels)

##############LOG REG################
"""
def logreg_obj(v, DTR, LTR, l):
    w, b = v[0:-1], v[-1]

    w_norm = numpy.linalg.norm(w)

    J = 0

    for i in range(DTR.shape[1]):

        if (LTR[i] == 1):
            z_i = 1
        else:
            z_i = -1

        J = J + numpy.logaddexp(0, -z_i * (numpy.dot(w.T, DTR[:, i]) + b))

    J = (J / DTR.shape[1]) + (l / 2) * w_norm ** 2

    return J

"""
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
    x0 = numpy.zeros(DTR.shape[0] + 1)

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, lamb,pi_t), approx_grad=True)

    w, b = x[0:-1], x[-1]

    S = []

    for i in range(DTE.shape[1]):
        S.append(numpy.dot(w.T, DTE[:, i]) + b)

    return S #np.dot(w.T,DTE)+b

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
            scores = binaryLogisticRegression(DTR, LTR, DTE, l)
            orderedLabels=LTE
        """
        #predLabels = np.array(np.array(scores) > 0).astype(int)
        #Conf = calculateConfusionMatrix(predLabels, LTE)
        dcf1.append(computeMinDCF(0.1, 1, 1, scores, orderedLabels))
        dcf5.append(computeMinDCF(0.5, 1, 1, scores, orderedLabels))
        dcf9.append(computeMinDCF(0.9, 1, 1, scores, orderedLabels))

    return (np.vstack([dcf1,dcf5,dcf9]),index)


def scheduleTask(mainTask,workingVet,DTR,LTR,DTE,LTE):
    resVet=[]
    resdict={}
    nWorkers= os.cpu_count()
    print('number of workers: ',nWorkers)
    start=time.time()
    with ProcessPoolExecutor(max_workers=nWorkers) as executor:
        #videoList = os.listdir(datasetPath)[0:10]

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

def main():
    PCA_flag=True
    PCA_factor=6
    LDA_flag=False
    GC_flag=False
    NBGC_flag=False
    TCGC_flag=False
    TCNBGC_flag=False

    logRegFlag=True
    
    priors=[0.5,0.9,0.1]
    
    #load of training and test set with labels
    (DTR, LTR), (DTE, LTE)=loadPulsar()
    
    #Z normalization of data
    DTR, meanDTR, standardDeviationDTR=ZNormalization(DTR)
    DTE, meanDTE, standardDeviationDTE=ZNormalization(DTE)

    #principal component analisys
    if PCA_flag:
        DTR=PCA(DTR,PCA_factor)
        DTE=PCA(DTE,PCA_factor)
        
    #linear discriminant analisis
    if LDA_flag:
        
        W=LDA(DTR,LTR,1)
        
        DTR=numpy.dot(W.T,DTR)
        DTE=numpy.dot(W.T,DTE)

    MVGres=np.empty((4,3))
    #gaussian classifier
    if GC_flag:
        (SPost,orderedLabels)=GC_KFold(DTR,LTR)
        predLabels = numpy.argmax(numpy.exp(SPost), 0)
        correct,total,precision=calculatePrecision(predLabels, orderedLabels)
        Conf=calculateConfusionMatrix(predLabels,orderedLabels)
        print("The Gaussian classifier classifies correctly",correct,"labels over", total, "total test samples. The precision is", precision*100, "%, the error rate is ", (1-precision)*100, "%")
        print("Confusion matrix:")
        print(Conf)
        for i,prior in enumerate(priors):
            #DCF=computeDCF(Conf,1,1,prior)
            llr = SPost[1, :] - SPost[0, :]
            DCF = computeMinDCF(prior, 1, 1, llr, orderedLabels)
            Bdummy=computeBDummy(1,1,prior)
            print("For prior probability: pi=",prior)
            print("Empirical Bayes risk:",DCF)
            MVGres[0,i]=DCF
            print("Normalized DCF:",(DCF/Bdummy))
       
    #naive bayes gaussian classifier
    if NBGC_flag:
        (SPost,orderedLabels)=NBGC_KFold(DTR,LTR)
        predLabels=numpy.argmax(numpy.exp(SPost),0)
        correct,total,precision=calculatePrecision(predLabels, orderedLabels)
        Conf=calculateConfusionMatrix(predLabels,orderedLabels)
        print("The Naive-Bayes Gaussian classifier classifies correctly",correct,"labels over", total, "total test samples. The precision is", precision*100, "%, the error rate is ", (1-precision)*100, "%")
        print("Confusion matrix:")
        print(Conf)
        for i,prior in enumerate(priors):
            #DCF=computeDCF(Conf,1,1,prior)
            llr=SPost[1,:]-SPost[0,:]
            DCF=computeMinDCF(prior,1,1,llr,orderedLabels)
            Bdummy=computeBDummy(1,1,prior)
            print("For prior probability: pi=",prior)
            print("Empirical Bayes risk:",DCF)
            MVGres[1, i] = DCF
            print("Normalized DCF:",(DCF/Bdummy))
        
    #tied covariance gaussian classifier
    if TCGC_flag:
        (SPost,orderedLabels)=TCGC_KFold(DTR,LTR)
        predLabels = numpy.argmax(numpy.exp(SPost), 0)
        correct,total,precision=calculatePrecision(predLabels, orderedLabels)
        Conf=calculateConfusionMatrix(predLabels,orderedLabels)
        print("The Tied Covariance Gaussian classifier classifies correctly",correct,"labels over", total, "total test samples. The precision is", precision*100, "%, the error rate is ", (1-precision)*100, "%")
        print("Confusion matrix:")
        print(Conf)
        for i,prior in enumerate(priors):
            #DCF=computeDCF(Conf,1,1,prior)
            llr = SPost[1, :] - SPost[0, :]
            DCF = computeMinDCF(prior, 1, 1, llr, orderedLabels)
            Bdummy=computeBDummy(1,1,prior)
            print("For prior probability: pi=",prior)
            print("Empirical Bayes risk:",DCF)
            MVGres[2, i] = DCF
            print("Normalized DCF:",(DCF/Bdummy))
            
    #tied covariance naive bayes gaussian classifier
    if TCNBGC_flag:
        (SPost,orderedLabels)=TCNBGC_KFold(DTR,LTR)
        predLabels = numpy.argmax(numpy.exp(SPost), 0)
        correct,total,precision=calculatePrecision(predLabels, orderedLabels)
        Conf=calculateConfusionMatrix(predLabels,orderedLabels)
        print("The Tied Covariance Naive-Bayes Gaussian classifier classifies correctly",correct,"labels over", total, "total test samples. The precision is", precision*100, "%, the error rate is ", (1-precision)*100, "%")
        print("Confusion matrix:")
        print(Conf)
        for i,prior in enumerate(priors):
            #DCF=computeDCF(Conf,1,1,prior)
            llr = SPost[1, :] - SPost[0, :]
            DCF = computeMinDCF(prior, 1, 1, llr, orderedLabels)
            Bdummy=computeBDummy(1,1,prior)
            print("For prior probability: pi=",prior)
            print("Empirical Bayes risk:",DCF)
            MVGres[3, i] = DCF
            print("Normalized DCF:",(DCF/Bdummy))

        if PCA_flag:print('MVG PCA: ',PCA_factor)
        else:print('MVG NO PCA')
        print('Full-Cov: ',MVGres[0])
        print('Diag-Cov: ', MVGres[1])
        print('Tied-Cov: ', MVGres[2])
        print('Diag-Tied-Cov: ', MVGres[3])


if __name__ == '__main__':
    main()
