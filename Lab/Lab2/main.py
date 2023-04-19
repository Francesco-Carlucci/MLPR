# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

DatasetPath='../2_soluzioni/iris.csv'

def col(v):
    return v.reshape((v.size,1))

def load(file):
    DList=[]
    labels=[]
    hLabels = {'Iris-setosa': 0,
               'Iris-versicolor':1,
               'Iris-virginica': 2}

    with open(file) as fp:
        #reader = csv.reader(fp, delimiter=',')
        #for row in reader:
        for row in fp:
            try:
                attrs=row.split(',')[0:4] #select features
                attrs=col(np.array([float(i) for i in attrs])) #converts to float, to numpy array, then to column array
                name=row.split(',')[-1].strip() #read name of the label, strip removes newline
                label=hLabels[name]  #convert to int label
                DList.append(attrs)
                labels.append(label)
            except:
                pass
    return np.hstack(DList), np.array(labels,dtype=np.int32)

def load2():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

if __name__ == '__main__':
    D,L=load(DatasetPath)
    D0=D[:,L==0] #select all element of class 0, using a mask from L (L==0)
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]
    hFea={
        0:'Sepal Length',
        1:'Sepal Width',
        2:'Petal Length',
        3:'Petal Width'
    }
    #print(D0.shape)
    #plt.figure()
    fig,axs=plt.subplots(2,2)
    for dIdx in range(4):
        axs[1 if dIdx<2 else 0,dIdx%2].set_xlabel(hFea[dIdx])
        axs[1 if dIdx<2 else 0,dIdx%2].hist(D0[dIdx,:],bins=10, density=True,alpha = 0.4, label = 'Setosa')
        axs[1 if dIdx<2 else 0,dIdx%2].hist(D1[dIdx,:], bins=10, density=True,alpha=0.4,label= 'Versicolor' )
        axs[1 if dIdx<2 else 0,dIdx%2].hist(D2[dIdx,:], bins=10, density=True,alpha=0.4,label='Virginica' )

        plt.legend()
        plt.tight_layout()
    plt.show()


    #vd. hist plotting
