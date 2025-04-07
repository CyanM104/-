import numpy as np
import matplotlib.pyplot as plt

def two_point_fwd_diff(f,x,h):
    res=f(x+h)-f(x)
    res/=h
    return res
def two_point_bwd_diff(f,x,h):
    res=f(x)-f(x-h)
    res/=h
    return res
def three_point_diff(f,x,h):
    res=f(x+h)-f(x-h)
    res/=2*h
    return res
def second_deriv(f,x,h):
    res=(f(x+h)-2*f(x)+f(x-h))
    res/=h**2
    return res