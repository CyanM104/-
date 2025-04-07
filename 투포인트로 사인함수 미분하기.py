import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-np.pi,np.pi,100)
h=0.0000001
fwd_dfdx=np.zeros_like(x)
bwd_dfdx=np.zeros_like(x)

for i in range(len(x)):
    fwd_dfdx[i]= (np.sin(x[i]+h)-np.sin(x[i]))/h
    bwd_dfdx[i]= (np.sin(x[i])-np.sin(x[i]-h))/h

know_sol=np.cos(x)

plt.plot(x,fwd_dfdx,'ro',label='Forward Difference', mfc='none')
plt.plot(x,bwd_dfdx,'bo',label='Backward Difference', mfc='none')
plt.plot(x,know_sol,'g-',label='Analytical Solution', lw=2)
plt.legend()
plt.show()