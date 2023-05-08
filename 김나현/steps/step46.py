if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
from dezero import Variable, Model, optimizers
from dezero.models import MLP
import dezero.functions as F
import dezero.layers as L

np.random.seed(0)
x=np.random.rand(100,1)
y=np.sin(2*np.pi*x)+np.random.rand(100,1)

lr=0.2
max_iter=10000
hidden_size=10

model=MLP((hidden_size, 1))
optimizer=optimizers.SGD(lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred=model(x)
    loss=F.mean_squared_error(y,y_pred)

    model.cleargrads()
    loss.backward()

    """for p in model.params():
        p.data-=lr*p.grad.data"""
    optimizer.update()
    if i%1000==0:
        print(loss)
