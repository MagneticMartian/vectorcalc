
import numpy as np
import numdifftools as nd
from itertools import permutations

class VectCalc(object):
    def __init__(self,f,x):
        self.f = f
        self.x = x

    def div(self):
        return np.trace(nd.Jacobian(self.f)(self.x))

    def grad(self):
        return nd.Gradient(self.f)(self.x)

    def laplacian(self):
        return sum(nd.Hessdiag(self.f)(self.x))

    def curl(self):
        n = len(self.x)
        prod = np.zeros(n)
        jac = nd.Jacobian(self.f)(self.x)
        indx = permutations(range(n))
        for i in indx:
            prod[i[0]] += self.levi_civita([i[0],i[1],i[2]])*jac[i[2],i[1]]
        return prod

    def levi_civita(self,indx):
        if(len(set(indx))==len(indx)):
            parity = 1
            for i in range(0,len(indx)-1):
                if (indx[i] != i):
                    parity *= -1
                    mn = min(range(i,len(indx)), key=indx.__getitem__)
                    indx[i],indx[mn] = indx[mn],indx[i]
            return parity
        else:
            return 0