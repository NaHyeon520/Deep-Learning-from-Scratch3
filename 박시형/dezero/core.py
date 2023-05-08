import numpy as np
import weakref
import contextlib
# import dezero.function as F
import dezero

## Config ##
class Config:
  enable_backprop = True
  
@contextlib.contextmanager
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config("enable_backprop", False)


## Variable ##
class Variable:
  __array_priority__ = 200
  def __init__(self, data, name=None) -> None:
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError(f'{type(data)}는 지원하지 않는 타입입니다.')
    self.data = data
    self.grad = None
    self.creator = None
    self.name = name
    self.generation = 0
    
  @property
  def shape(self):
    return self.data.shape
  
  @property
  def size(self):
    return self.data.size
  
  @property
  def dtype(self):
    return self.data.dtype
  
  @property
  def T(self):
    return dezero.function.transpose(self)
  
  def __len__(self):
    return len(self.data)
  
  def __repr__(self):
    if self.data is None:
      return "Variable(None)"
    else:
      p = str(self.data).replace("\n", "\n" + ' ' * 9)
      return f"Variable({p})"
  
  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    # return F.reshape(self, shape)
    return dezero.function.reshape(self, shape)
  
  def transpose(self):
    return dezero.function.transpose(self)
  
  def sum(self, axis, keepdims=False):
    return dezero.function.sum(self, axis, keepdims)
    
    
  def cleargrad(self):
    self.grad = None
  
  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1
  
  def backward(self, retain_grad=False, create_graph=False):
    if self.grad is None:
      self.grad = Variable(np.ones_like(self.data))
    
    funcs = []
    seen_set = set()
    def add_func(f):
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x: x.generation)
        
    add_func(self.creator)
    
    while len(funcs) != 0:
      f = funcs.pop()
      gys = [output().grad for output in f.outputs]
      
      with using_config("enable_backprop", create_graph):
        gxs = f.backward(*gys)
        if not isinstance(gxs, tuple):
          gxs = (gxs,)
        for x, gx in zip(f.inputs, gxs):
          if x.grad is None:
            x.grad = gx
          else:
            x.grad = x.grad + gx
          if x.creator is not None:
            add_func(x.creator)
          
      if not retain_grad:
        for y in f.outputs:
          y().grad = None


def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x


def as_variable(obj):
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)


## Parameter ##
class Parameter(Variable):
  pass

## Function ##
class Function:
  def __call__(self, *inputs):
    inputs = [as_variable(x) for x in inputs]
    xs = [x.data for x in inputs]
    ys = self.forward(*xs)
    if not isinstance(ys, tuple):
      ys = (ys,)
    outputs = [Variable(as_array(y)) for y in ys]
    
    if Config.enable_backprop:
      self.generation = max([x.generation for x in inputs])
      for output in outputs:
        output.set_creator(self)
      self.inputs = inputs
      self.outputs = [weakref.ref(output) for output in outputs]
    
    return outputs if len(outputs) > 1 else outputs[0]
  
  def forward(self, xs):
    raise NotImplementedError

  def backward(self, gys):
    raise NotImplementedError
  

## operation overloaing ##
class Add(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 + x1
  
  def backward(self, gy):
    gx0, gx1 = gy, gy
    if self.x0_shape != self.x1_shape:
      gx0 = dezero.function.sum_to(gx0, self.x0_shape)
      gx1 = dezero.function.sum_to(gx1, self.x1_shape)
      
    return gx0, gx1
  
class Mul(Function):
  def forward(self, x0, x1):
    return x0 * x1

  def backward(self, gy):
    x0, x1 = self.inputs   
    return gy * x1, gy * x0

class Neg(Function):
  def forward(self, x):
    return -x
  
  def backward(self, gy):
    return -gy

class Sub(Function):
  def forward(self, x0, x1):
    return x0 - x1
  
  def backward(self, gy):
    return gy, -gy
  
class Div(Function):
  def forward(self, x0, x1):
    return x0 / x1
  
  def backward(self, gy):
    x0, x1 = self.inputs
    gx0 = gy / x1
    gx1 = gy * (-x0 / x1 ** 2)
    return gx0, gx1
  

class Pow(Function):
  def __init__(self, c):
    self.c = c
  
  def forward(self, x):
    return x ** self.c

  def backward(self, gy):
    x, = self.inputs
    c = self.c
    gx = c * x ** (c-1) * gy
    return gx



def add(x0, x1):
  x1 = as_array(x1)
  return Add()(x0, x1)


def mul(x0, x1):
  x1 = as_array(x1)
  return Mul()(x0, x1)


def neg(x):
  return Neg()(x)


def sub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x0, x1)


def rsub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x1, x0)


def div(x0, x1):
  x1 = as_array(x1)
  return Div()(x0, x1)


def rdiv(x0, x1):
  x1 = as_array(x1)
  return Div()(x1, x0)


def pow(x, c):
  return Pow(c)(x)

def setup_variable():
  Variable.__mul__ = mul
  Variable.__rmul__ = mul

  Variable.__add__ = add
  Variable.__radd__ = add

  Variable.__neg__ = neg
  Variable.__sub__ = sub
  Variable.__rsub__ = rsub

  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv

  Variable.__pow__ = pow