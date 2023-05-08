if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from dezero import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
import dezero
import dezero.functions as F
from dezero.models import MLP
from dezero import optimizers
from dezero.datasets import Spiral
from dezero import DataLoader

batch_size=10
max_epoch=1

train_set=Spiral(train=True)
test_set=Spiral(train=False)
train_loader=DataLoader(train_set, batch_size)
test_loader=DataLoader(test_set, batch_size)

for epoch in range(max_epoch):
    for x,t in train_loader:
        print(x.shape, t.shape)
        break

    for x,t in test_loader:
        print(x.shape, t.shape)
        break

#50.3
y=np.array([[0.2,0.8,0],[0.1,0.9,0],[0.8,0.1,0.1]])
t=np.array([1,2,0])
acc=F.accuracy(y,t)
print(acc)

#50.4
batch_size=10
max_epoch=300
hidden_size=10
lr=0.1

train_set=Spiral(train=True)
test_set=Spiral(train=False)
train_loader=DataLoader(train_set, batch_size)
test_loader=DataLoader(test_set, batch_size, shuffle=False)

model=MLP((hidden_size, 3))
optimizer=optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc=0,0

    for x,t in train_loader:
        y=model(x)
        loss=F.softmax_cross_entropy(y,t)
        acc=F.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)
    
    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(train_set), sum_acc/len(train_set)))

    sum_loss, sum_acc=0,0
    with dezero.no_grad():
        for x,t in test_loader:
            y=model(x)
            loss=F.softmax_cross_entropy(y,t)
            acc=F.accuracy(y,t)
            sum_loss+=float(loss.data)*len(t)
            sum_acc+=float(acc.data)*len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(test_set), sum_acc/len(test_set)))