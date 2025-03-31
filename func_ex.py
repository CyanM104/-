from math import cos, sin, pi
from numpy import array
def factorial(n):
    f=1.0
    for i in range(1,n+1):
        f*=i
    return f

def combination(n,m):
    return factorial(n)/factorial(m)/factorial(n-m)

def my_sin(x,n_terms):
    sin=0.0
    for n in range(0,n_terms):
        order=2*n+1
        sin+=(-1)**n**order/factorial(order)
        return my_sin

def cartesian(r,theta):
    x=r*cos(theta)
    y=r*sin(theta)
    return array([x,y],float)

def f(x):
    return x**2