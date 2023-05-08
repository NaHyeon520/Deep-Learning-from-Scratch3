from ast import Return
import numpy as np
import weakref
import contextlib
import math

class Variable:
    __array_priority__=200
    
    def __init__(self, data, name=None):
      if data is not None:
          if not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))
      self.data=data
      self.name=name
      self.grad=None
      self.creator=None
      self.generation=0
    
    def set_creator(self,func):
      self.creator =func
      self.generation = func.generation +1

    def backward(self,retain_grad=False,create_graph=False):
      if self.grad is None: 
        self.grad=Variable(np.ones_like(self.data)) # 아무것도 없을 때 시작 값
      funcs = []
      seen_set =set()

      def add_func(f):  # 리스트를 세대 순으로 정렬
        if f not in seen_set:
          funcs.append(f)
          seen_set.add(f)
          funcs.sort(key=lambda x : x.generation) # funcs 리스트 요소 중에서 generation이 작은 거부터 큰 거 순으로 정렬
      
      add_func(self.creator) 
    
      while funcs:
        f= funcs.pop()
        gys = [output().grad for output in f.outputs] # f.outputs 리스트에 있는 모든 요소의 .grad 속성을 리스트로 추출하여 반환
        
        with using_config('enable_backprop',create_graph):
          gxs = f.backward(*gys)
          if not isinstance(gxs,tuple):
            gxs = (gxs,)

          for x,gx in zip(f.inputs,gxs):  # zip은 데이터를 묶어서 처리하는 함수
            if x.grad is None:
                x.grad = gx
            else :
                x.grad = x.grad +gx

            if x.creator is not None:
                add_func(x.creator)

          if not retain_grad:
            for y in f.outputs:
              y().grad = None  

    def cleargrad(self):
        self.grad=None

    def __len__(self):
      return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n','\n'+' '*9)
        return 'variable('+p+')'

    def __mul__(self, other):
        return mul(self, other)

    def reshape(self,*shape): # reshape 함수를 인스턴스 메서드 형태로 호출
      if len(shape) ==1 and isinstance(shape[0],(tuple.list)):
        shape =shape[0]
      return reshape(self,shape)  
    
    def transpose(self):
      return transpose(self)

    def sum(self,axis=None,keepdims=False):
      return sum(self,axis,keepdims)

    @property  #함수 호출 대신에 인스턴스 속성처럼 사용가능
    def shape(self):
      return self.data.shape
    @property
    def ndim(self):
      return self.data.ndim
    @property
    def dtype(self):
      return self.data.dtype
    @property
    def T(self):
      return transpose(self)


class Function:
    def __call__(self,*inputs): # *은 받은 값을 튜플 형태로 저장하는 역할
      inputs = [as_variable(x)for x in inputs]
      
      xs = [x.data for x in inputs]
      ys = self.forward(*xs)
      if not isinstance(ys,tuple): # 입력값이 튜플이 아니면 튜플로 바꿔주기
        ys = (ys,)
      outputs = [Variable(as_array(y)) for y in ys] # as_array는 주어진 값을 넘파이로 바꿔줌

      if Config.enable_backprop:
        self.generation = max([x.generation for x in inputs]) # inputs의 x.generation 중에서 가장 큰 값을 골라서 저장 
        for output in outputs :
          output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
    
      return outputs if len(outputs)> 1 else outputs[0]

    def forward(self,x):
      raise NotImplementedError() # 아직 구현하지 않은 부분에 대한 표식

    def backward(self,gy):
      raise NotImplementedError()

class Exp(Function):
    def forward(self,x):
      y= np.exp(x)
      return y

    def backward(self,gy):
      x=self.input.data
      gx=np.exp(x)*gy
      return gx

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0+x1
        return y
    
    def backward(self, gy):
        gx0,gx1=gy,gy
        if self.x0_shape != self.x1_shape:
          gx0= sum_to(gx0,self.x0_shape)
          gx1= sum_to(gx1,self.x1_shape)
        return gx0,gx1
    
class Mul(Function):
    def forward(self, x0, x1):
        y=x0*x1
        return y
    
    def backward(self, gy):
        x0,x1=self.inputs
        return gy*x1, gy*x0
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        y=x0-x1
        return y
    
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        y=x0/x1
        return y
    
    def backward(self, gy):
        x0,x1=self.inputs
        gx0=gy/x1
        gx1=gy*(-x0/x1**2)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c):
        self.c=c

    def forward(self, x):
        y=x**self.c
        return y
    
    def backward(self, gy):
        x,=self.inputs
        c=self.c
        gx=c*x**(c-1)*gy
        return gx

class Sin(Function):
  def forward(self,x):
    y = np.sin(x)
    return y
  def backward(self,gy):
    x,=self.inputs
    gx=gy*cos(x)
    return gx

class Cos(Function):
  def forward(self,x):
    y=np.cos(x)
    return y
  def backward(self,gy):
    x,=self.inputs
    gx=gy*-sin(x)
    return gx

class Tanh(Function):
  def forward(self, x):
    y = np.tanh(x)
    return y

  def backward(self, gy):
    y = self.outputs[0]()
    gx = gy * (1 - y * y)
    return gx

class Reshape(Function): # 텐서의 형태를 입력변수에 맞추어 변환함
  def __init__(self,shape):
    self.shape =shape

  def forward(self,x):
    self.x_shape=x.shape
    y=x.reshape(self.shape)
    return y

  def backward(self,gy):
    return reshape(gy,self.x_shape)

def reshape(x,shape):
  if x.shape==shape:
    return as_variable(x)
  return Reshape(shape)(x)

class Transpose(Function):
  def forward(self,x):
    y=np.transpose(x)
    return y
  def backward(self,gy):
    gx=transpose(gy)
    return gx

def transpose(x):
  return Transpose()(x)

class Sum(Function):
  def __init__(self,axis,keepdims):
    self.axis =axis
    self.keepdims =keepdims

  def forward(self,x):
    self.x_shape = x.shape
    y= x.sum(axis=self.axis,keepdims=self.keepdims)
    return y
  
  def backward(self,gy):
    gy = reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
    
    gx = broadcast_to(gy,self.x_shape)
    return gx

def sum(x,axis=None,keepdims=False):
  return Sum(axis,keepdims)(x)

class MatMul(Function):
  def forward(self,x,W):
    y= x.dot(W)
    return y

  def backward(self,gy):
    x,W=self.inputs
    gx=matmul(gy,W.T)
    gW=matmul(x.T,gy)
    return gx,gW

def matmul(x,W):
  return MatMul()(x,W)

class BroadcastTo(Function):
  def __init__(self,shape):
    self.shape=shape

  def forward(self,x):
    self.x_shape =x.shape
    y=np.broadcast_to(x,self.shape)
    return y

  def backward(self,gy):
    gx= sum_to(gy,self.x_shape)
    return gx

def broadcast_to(x,shape):
  if x.shape ==shape:
    return as_variable(x)
  return BroadcastTo(shape)(x)

class SumTo(Function):
  def __init__(self,shape):
    self.shape =shape

  def forward(self,x):
    self.x_shape =x.shape
    y=sum_to(x,self.shape)
    return y

  def backward(self,gy):
    gx=broadcast_to(gy,self.x_shape)
    return gx

def sum_to(x,shape):
  if x.shape==shape:
    return as_variable(x)
  return SumTo(shape)(x)

def exp(x):
    return Exp()(x)
def add(x0, x1):
    x1=as_array(x1)
    return Add()(x0,x1)
def mul(x0, x1):
    x1=as_array(x1)
    return Mul()(x0, x1)
def neg(x):
  return Neg()(x)
def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0, x1)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1, x0)
def div(x0,x1):
  x1=as_array(x1)
  return Div()(x0,x1)
def rdiv(x0,x1):
  x1=as_array(x1)
  return Div()(x1,x0)
def pow(x,c):
  return Pow(c)(x)

Variable.__mul__=mul # 연산자 오버로딩
Variable.__rmul__=mul
Variable.__add__=add
Variable.__radd__=add  
Variable.__neg__=neg
Variable.__sub__=sub
Variable.__rsub__=rsub
Variable.__truediv__=div
Variable.__rtruediv__=rdiv
Variable.__pow__=pow

#삼각함수
def sin(x):
  return Sin()(x)
def cos(x):
  return Cos()(x)
def tanh(x):
  return Tanh()(x)


def numerical_diff(f,x,eps=1e-4):
  x0=Variable(x.data-eps)
  x1=Variable(x.data+eps)
  y0=f(x0)
  y1=f(x1)
  return (y1.data-y0.data)/(2*eps)

class Config: # 역전파 활성화
  enable_backprop = True

@contextlib.contextmanager
def using_config(name,value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config('enable_backprop',False)

def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

def as_variable(obj):
  if isinstance(obj,Variable):
    return obj
  return Variable(obj)

def sphere(x,y):
  z=x**2+y**2
  return z

def matyas(x,y):
  z=0.26*(x**2+y**2)-0.48*x*y
  return z

def goldstein(x,y):
  z=(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
  return z

def my_sin(x, threshold=0.0001): # 판단을 내리는데 필요한 임계값
  y=0
  for i in range(100000):
    c=(-1)**i/math.factorial(2*i+1) # 테일러 급수 계수 유도
    t=c*x**(2*x+i)
    y=y+t
    if abs(t.data) < threshold:
      break
  return y

def rosenbrock(x0,x1):
  y= 100*(x1-x0**2)**2+(1-x0)**2
  return y

def f(x):
  y= x**4-2*x**2
  return y

def gx2(x):
  return 12*x**2-4


np.random.seed(0)
x=np.random.rand(100,1)
y=5+2*x+np.random.rand(100,1)
x,y=Variable(x),Variable(y)

W= Variable(np.zeros((1,1)))
b= Variable(np.zeros(1))

def predict(x):
  y=matmul(x,W)+b
  return y

def mean_squared_error(x0,x1):
  diff=x0-x1
  return sum(diff**2)/len(diff)
