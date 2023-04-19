import load
import numpy as np
import time
from scipy import special

def vcol(v):
    return v.reshape(v.size,1)
def vrow(v):
    return v.reshape(1,v.size)

def computeDictionary(lTrain,nClasses):
    D = {}
    cnt = 0
    for i in range(nClasses):
        for tercet in lTrain[i]:
            for word in tercet.split():
                if word not in D.keys():
                    D[word] = cnt
                    cnt += 1
    return D

def computeOccs(D,text,eps):
    occCnt =np.zeros(len(D.keys()))+eps
    for tercet in text:
        for word in tercet.split():
            if word in D.keys():
                occCnt[D[word]] += 1
    return occCnt

def testOcc(lTest):
    y=[]
    for tercet in lTest:
        y.append(vcol(computeOccs(D, [tercet],0)))

    return np.hstack(y)

def printAccuracy(accuracy,LTE):
    errRate = 1 - accuracy
    print('accuracy: ', accuracy)
    print('error rate: %.3f' % errRate)

def calculateAccfromPost(SPost,LTE):
    predLabels = np.argmax(SPost, axis=0)
    correct = np.sum(predLabels == LTE)
    accuracy = correct / predLabels.shape[0]
    return accuracy
    #printAccuracy(predLabels,LTE)

if __name__=='__main__':
    linf,lpur,lpar=load.load_data()
    nClasses=3

    linfTrain,linfTest=load.split_data(linf,4)
    lpurTrain, lpurTest = load.split_data(lpur, 4)
    lparTrain, lparTest = load.split_data(lpar, 4)

    lTrain=[linfTrain,lpurTrain,lparTrain]
    N_c=[len(lTrain[i]) for i in range(nClasses)]
    #Ninf=len(linfTrain)
    #Npur=len(lpurTrain)
    #Npar=len(lparTrain)

    eps=0.001
    """
    start1=time.time()
    strTrain=''
    for i in range(nClasses):
        for j in range(N_c[i]):
            strTrain = strTrain +" "+ lTrain[i][j]

    D=set(strTrain.split())
    print('1° metodo: ',time.time()-start1)

    start1=time.time()
    D2={}
    occCnt=[[] for i in range(nClasses)]
    cnt=0
    for i in range(nClasses):
        occCnt[i]=np.zeros(len(D2.keys()))
        for tercet in lTrain[i]:
            for word in tercet.split():
                if word not in D2.keys():
                    D2[word]=cnt
                    cnt+=1
                    occCnt[i]=np.append(occCnt[i],1)
                else:
                    occCnt[i][D2[word]]+=1
    padLength=len(D2.keys())
    for i in range(nClasses-1):
        occCnt[i]=np.pad(occCnt[i],[(0,padLength-len(occCnt[i]))],mode='constant')
    print('1° metodo: ', time.time() - start1)
    """
    #start2=time.time()

    D=computeDictionary(lTrain,nClasses)
    occCnt=[]
    for i in range(nClasses):
        occCnt.append(computeOccs(D,lTrain[i],eps))

    #print('2° metodo',time.time() - start2)

    normOcc=[occCnt[i]/occCnt[i].sum() for i in range(nClasses)]
    w_c=np.log(normOcc)

    #compute class conditional log-likelihood

    y=testOcc(linfTest)
    Sinf=np.array([vrow(np.dot(y.T,vcol(w_c[i])))[0] for i in range(nClasses)])
    SJoint=Sinf+np.log([[1.0/3.0]])
    SMarginal=special.logsumexp(SJoint,axis=0)
    SPost=SJoint-SMarginal

    infAcc=calculateAccfromPost(np.exp(SPost),[[0]])
    print('Multiclass accuracy Inferno: ',infAcc)

    y = testOcc(lpurTest)
    Spur = np.array([vrow(np.dot(y.T, vcol(w_c[i])))[0] for i in range(nClasses)])
    SJoint = Spur + np.log([[1.0 / 3.0]])
    SMarginal = special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SMarginal

    purAcc=calculateAccfromPost(np.exp(SPost),[[1]])
    print('Multiclass accuracy Purgatorio: ', purAcc)

    y = testOcc(lparTest)
    Spar = np.array([vrow(np.dot(y.T, vcol(w_c[i])))[0] for i in range(nClasses)])
    SJoint = Spar + np.log([[1.0 / 3.0]])
    SMarginal = special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SMarginal

    parAcc = calculateAccfromPost(np.exp(SPost), [[2]])
    print('Multiclass accuracy Paradiso: ', parAcc)

    SInfB=np.vstack([Sinf[0:1,:],Sinf[2:3,:]])
    SParB=np.vstack([Spar[0:1,:],Spar[2:3,:]])
    SB=np.hstack([SInfB,SParB])

    infLabels=[0 for i in range(SInfB.shape[1])]
    parLabels=[1 for i in range(SParB.shape[1])]
    BLabels=np.hstack([infLabels,parLabels])
    BAcc=calculateAccfromPost(SB,BLabels)
    print('Binary accuracy inferno/paradiso: ',BAcc)

    SInfB = np.vstack([Sinf[0:1, :], Sinf[1:2, :]])
    SPurB = np.vstack([Spur[0:1, :], Spur[1:2, :]])
    SB = np.hstack([SInfB, SPurB])

    #infLabels = [0 for i in range(SInfB.shape[1])]
    purLabels = [1 for i in range(SPurB.shape[1])]
    BLabels = np.hstack([infLabels, purLabels])
    BAcc = calculateAccfromPost(SB, BLabels)
    print('Binary accuracy inferno/purgatorio: ', BAcc)

    SPurB = np.vstack([Spur[1:2, :], Spur[2:3, :]])
    SParB = np.vstack([Spar[1:2, :], Spar[2:3, :]])
    SB = np.hstack([SPurB, SParB])

    purLabels = [0 for i in range(SPurB.shape[1])]
    parLabels = [1 for i in range(SParB.shape[1])]
    BLabels = np.hstack([purLabels, parLabels])
    BAcc = calculateAccfromPost(SB, BLabels)
    print('Binary accuracy purgatorio/paradiso: ', BAcc)