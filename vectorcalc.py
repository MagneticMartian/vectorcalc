import numpy as np                 #For array and vectorization ease
import numdifftools as nd          #For the numerical analysis functions
from itertools import permutations #For implementation with levi_civita

"""
This class is designed for simplicity of implementation.
It is designed to be as light weight and efficient as
possible without making sacrafices to hardcoding. The 
primary purpose is to fill a precieved gap in the amount
of classes and software packages that do not implement 
the most basic and fundamental of Vector Analysis
operations.
"""

class VectCalc(object):
    def __init__(self,f,x):
        self.f = f #provided function template
        self.x = x #provided vector space definition

    def div(self):
        """
        The divergence of a vector space is a measure of
        how quickly it spreads away from the spaces origin.
        In the cartesian coordinate system the divergence is

        div(f) = (D_x X + D_y Y + D_z Z +...).(f_x X + f_y Y + f_z Z + ...)
               = D_x f_x + D_y f_y + D_z f_z + ...
        """
        return np.trace(nd.Jacobian(self.f)(self.x))

    def grad(self):
        """
        The gradiant of a scalar field is a measure of the
        direction of steepest decient. It is deinfed by

        grad(f) = D_x f + D_y f + D_z f + ...

        This is the one function that is already prebuilt
        into the numdifftools lib. This wrapper was simply
        created to improve ease of use for the function.
        """
        return nd.Gradient(self.f)(self.x)

    def laplacian(self):
        """
        The laplacian for a scalar field is defined as the
        divergence of the of the gradiant of the scalar
        field, and is written in cartesian coordinates

        Lap(f) = D^2_x f + D^2_y f + D^2_z f + ...
        """
        return sum(nd.Hessdiag(self.f)(self.x))

    def curl(self):
        """
        """
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