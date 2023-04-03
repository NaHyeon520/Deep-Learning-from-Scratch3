if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt

def rosenbrock(x0, x1):
    y=100*(x1-x0**2)**2+(1-x0)**2
    return y

x0=Variable(np.array(0.0))
x1=Variable(np.array(2.0))
lr=0.001
iters=10000
X0=[x0.data.astype(float)]
X1=[x1.data.astype(float)]

for i in range(iters):
    print(x0,x1)
    y=rosenbrock(x0,x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    x0.data-=lr*x0.grad
    x1.data-=lr*x1.grad
    X0.append(x0.data.astype(float))
    X1.append(x1.data.astype(float))


plt.scatter(1.0, 1.0)
plt.plot(X0, X1, 'ro')

plt.show()