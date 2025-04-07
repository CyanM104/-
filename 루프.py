import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    f=1.0
    for i in range(1,n+1):
        f*=i
    return f

def sin(x,n):
    result=np.zeros_like(np.array(x))
    for i in range(0,n+1):
        result+=(-1)**i*x**(2*i+1)/factorial(2*i+1)
    return result

x=np.linspace(-np.pi,np.pi,100)
num_terms=5
y=np.array(len(x),float)

for n in range(0,num_terms,1):
    y=sin(x,n)
    plt.plot(x,y,label=r'$n=$%d' %(n+1))
    plt.legend()
plt.plot(x,np.sin(x),'ro',mfc='none')
plt.xlim(-np.pi,np.pi)
plt.ylim(-1.5,1.5)
plt.show()