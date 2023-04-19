import scipy.optimize
import numpy as np

def Gradf(x):
    y = x[0]
    z = x[1]

    dy=2*(y+3)+np.cos(y)
    dz=2*(z+1)

    return np.array([dy,dz])

def f(x):
    y=x[0]
    z=x[1]

    res=(y+3)**2+np.sin(y)+(z+1)**2

    return res

def f_grad(x):

    res=f(x)
    grad=Gradf(x)
    return res,grad

if __name__=='__main__':
    x0=np.array([0,0])
    x,min,d=scipy.optimize.fmin_l_bfgs_b(f,x0,approx_grad=True)

    print('minimum without gradient: ',x,min,d)

    x2, min2, d2 = scipy.optimize.fmin_l_bfgs_b(f_grad, x0)

    print('minimum with gradient: ', x2, min2, d2)
