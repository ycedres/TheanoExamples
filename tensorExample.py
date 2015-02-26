__author__ = 'ycedres'

import theano
import theano.tensor as T
import numpy as np

W = T.matrix('W')
x = T.vector('x')
b = T.vector('b')

y = T.dot(W,x) + b

linear_out = theano.function([W,x,theano.Param(b,default=np.array([0,0]))],[y])

pesos = np.array([[1,2,3],[4,5,6]])
input = np.array([1,2,3])

print linear_out(pesos,input)