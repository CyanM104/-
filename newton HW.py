import numpy as np

def f(x):
    return np.exp(x)*np.log(x)-x**2
def dfdx(x):
    return np.exp(x)*np.log(x)+np.exp(x)/x-2*x
def g(x):
    return x**5-3*x**4-5*x**3+x**2+x+3
def dgdx(x):
    return 5*x**4-12*x**3-15*x**2+2*x+1
def dfdx_num(func,x0,x1):
    return (func(x1)-func(x0))/(x1-x0)

tolerance=1.0e-8
a=1.0
a0=1.5
b=2.0
b0=2.5

while 1:
    deltaa=-f(a)/dfdx(a)
    a+=deltaa
    if np.abs(deltaa)<tolerance:
        break
print("a=%f" %(a))
while 1:
    deltab=-f(b)/dfdx(b)
    b+=deltab
    if np.abs(deltab)<tolerance:
        break
print("b=%f" %(b))
while 1:
    deltaa=-f(a)/dfdx_num(f,a0,a)
    a+=deltaa
    if np.abs(deltaa)<tolerance:
        break
print("a=%f" %(a))
while 1:
    deltab=-f(b)/dfdx_num(f,b0,b)
    b+=deltab
    if np.abs(deltab)<tolerance:
        break
print("b=%f" %(b))


while 1:
    deltaa=-g(a)/dgdx(a)
    a+=deltaa
    if np.abs(deltaa)<tolerance:
        break
print("a=%f" %(a))
while 1:
    deltab=-g(b)/dgdx(b)
    b+=deltab
    if np.abs(deltab)<tolerance:
        break
print("b=%f" %(b))
while 1:
    deltaa=-g(a)/dfdx_num(g,a0,a)
    a+=deltaa
    if np.abs(deltaa)<tolerance:
        break
print("a=%f" %(a))
while 1:
    deltab=-g(b)/dfdx_num(g,b0,b)
    b+=deltab
    if np.abs(deltab)<tolerance:
        break
print("b=%f" %(b))