import numpy as np
from matplotlib import pyplot as plt

def makecol(vet):
    return vet.reshape(vet.size,1)

def loadIris(fileName):
    labelList=[]
    attrList=[]
    labeltoIdx={'Iris-setosa':0,
                'Iris-versicolor':1,
                'Iris-virginica':2}
    #try:
    with open(fileName) as f:
        for line in f:
            *attr,label=line.strip().split(sep=',')
            attr=np.array([float(i) for i in attr])
            labelList.append(labeltoIdx[label])
            attrList.append(attr.reshape(attr.size,1))
    attrList=np.hstack(attrList)
    labelList=np.array(labelList)
    #except:
        #print('Error opening file')
        #exit(1)
    return attrList,labelList

def drawHist(data0,data1,data2):

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(data0[0, :], bins=10, alpha=0.5)
    axs[0, 0].hist(data1[0, :], bins=10, alpha=0.5)
    axs[0, 0].hist(data2[0, :], bins=10, alpha=0.5)
    axs[0, 0].legend(['Setosa', 'Versicolor', 'Virginica'])
    axs[0, 0].set_xlabel('Sepal Length')
    # plt.figure()
    axs[0, 1].hist(data0[1, :], bins=10, alpha=0.5)
    axs[0, 1].hist(data1[1, :], bins=10, alpha=0.5)
    axs[0, 1].hist(data2[1, :], bins=10, alpha=0.5)
    axs[0, 1].legend(['Setosa', 'Versicolor', 'Virginica'])
    axs[0, 0].set_xlabel('Sepal Width')
    # plt.figure()
    axs[1, 0].hist(data0[2, :], bins=10, alpha=0.5)
    axs[1, 0].hist(data1[2, :], bins=10, alpha=0.5)
    axs[1, 0].hist(data2[2, :], bins=10, alpha=0.5)
    axs[1, 0].legend(['Setosa', 'Versicolor', 'Virginica'])
    axs[0, 0].set_xlabel('Petal Length')
    # plt.figure()
    axs[1, 1].hist(data0[3, :], bins=10, alpha=0.5)
    axs[1, 1].hist(data1[3, :], bins=10, alpha=0.5)
    axs[1, 1].hist(data2[3, :], bins=10, alpha=0.5)
    axs[1, 1].legend(['Setosa', 'Versicolor', 'Virginica'])
    axs[0, 0].set_xlabel('Petal Width')
    #plt.show()

def drawScatter(data0,data1,data2):
    #plt.figure()
    fig, axs = plt.subplots(3,3)
    #[a.legend(['Setosa', 'Versicolor', 'Virginica']) for a in axs.ravel()]

    axs[0,0].scatter(data0[0,:],data0[1,:])
    axs[0,0].scatter(data1[0, :],data1[1, :])
    axs[0,0].scatter(data2[0, :],data2[1, :])
    axs[0,0].set_xlabel('Sepal length')
    axs[0,0].set_ylabel('Sepal width')
    axs[0,0].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[0, 1].scatter(data0[0, :], data0[2, :])
    axs[0, 1].scatter(data1[0, :], data1[2, :])
    axs[0, 1].scatter(data2[0, :], data2[2, :])
    axs[0, 1].set_ylabel('Sepal length')
    axs[0, 1].set_xlabel('Petal length')
    axs[0, 1].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[0, 2].scatter(data0[0, :], data0[3, :])
    axs[0, 2].scatter(data1[0, :], data1[3, :])
    axs[0, 2].scatter(data2[0, :], data2[3, :])
    axs[0, 2].set_ylabel('Sepal length')
    axs[0, 2].set_xlabel('Petal width')
    axs[0, 2].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[1, 0].scatter(data0[1, :], data0[0, :])
    axs[1, 0].scatter(data1[1, :], data1[0, :])
    axs[1, 0].scatter(data2[1, :], data2[0, :])
    axs[1, 0].set_ylabel('Sepal width')
    axs[1, 0].set_xlabel('Sepal length')
    axs[1, 0].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[1, 1].scatter(data0[1, :], data0[2, :])
    axs[1, 1].scatter(data1[1, :], data1[2, :])
    axs[1, 1].scatter(data2[1, :], data2[2, :])
    axs[1, 1].set_ylabel('Sepal width')
    axs[1, 1].set_xlabel('Petal length')
    axs[1, 1].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[1, 2].scatter(data0[1, :], data0[3, :])
    axs[1, 2].scatter(data1[1, :], data1[3, :])
    axs[1, 2].scatter(data2[1, :], data2[3, :])
    axs[1, 2].set_ylabel('Sepal width')
    axs[1, 2].set_xlabel('Petal width')
    axs[1, 2].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[2, 0].scatter(data0[2, :], data0[0, :])
    axs[2, 0].scatter(data1[2, :], data1[0, :])
    axs[2, 0].scatter(data2[2, :], data2[0, :])
    axs[2, 0].set_ylabel('Petal length')
    axs[2, 0].set_xlabel('Sepal length')
    axs[2, 0].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[2, 1].scatter(data0[2, :], data0[1, :])
    axs[2, 1].scatter(data1[2, :], data1[1, :])
    axs[2, 1].scatter(data2[2, :], data2[1, :])
    axs[2, 1].set_ylabel('Petal length')
    axs[2, 1].set_xlabel('Sepal width')
    axs[2, 1].legend(['Setosa', 'Versicolor', 'Virginica'])

    axs[2, 2].scatter(data0[2, :], data0[3, :])
    axs[2, 2].scatter(data1[2, :], data1[3, :])
    axs[2, 2].scatter(data2[2, :], data2[3, :])
    axs[2, 2].set_ylabel('Petal length')
    axs[2, 2].set_xlabel('Petal width')
    axs[2, 2].legend(['Setosa', 'Versicolor', 'Virginica'])
    #plt.show()

if __name__=='__main__':
    data,labels=loadIris('iris.csv')
    data0 = data[:, labels == 0]  # select column of the 1 class
    data1 = data[:, labels == 1]
    data2 = data[:, labels == 2]
    drawHist(data0,data1,data2)
    drawScatter(data0,data1,data2)

    mean=data.mean(axis=1)
    #centering data
    cData=data-mean.reshape((data.shape[0],1))   #subtract the mean as a column
    cData0 = cData[:, labels == 0]  # select column of the 1 class
    cData1 = cData[:, labels == 1]
    cData2 = cData[:, labels == 2]
    drawHist(cData0, cData1, cData2)
    drawScatter(cData0, cData1, cData2)

    plt.show()



