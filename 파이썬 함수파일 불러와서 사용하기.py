import func_ex
import numpy as np
n=func_ex.factorial(10)
print(n)
y=func_ex.combination(5,2)
print(y)
print(func_ex.cartesian(10.2,np.pi/4.0))
x=list(range(-5,6))
z=list(map(func_ex.f,x))
print(z)