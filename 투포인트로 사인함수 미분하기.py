import numpy as np
import matplotlib.pyplot as plt
import my_module as mm

x=np.linspace(-np.pi,np.pi,100)
h=0.1
fwd_dfdx=np.zeros_like(x)
bwd_dfdx=np.zeros_like(x)
three_dfdx=np.zeros_like(x)
second_deriv=np.zeros_like(x)

fwd_dfdx=mm.two_point_fwd_diff(np.sin,x,h)
bwd_dfdx=mm.two_point_bwd_diff(np.sin,x,h)
three_dfdx=mm.three_point_diff(np.sin,x,h)
second_deriv=mm.second_deriv(np.sin,x,h)

plt.plot(x,fwd_dfdx,'ro',label='Forward Difference', mfc='none')
plt.plot(x,bwd_dfdx,'bo',label='Backward Difference', mfc='none')
plt.plot(x,three_dfdx,'go',label='Three-Point Difference', mfc='none')
plt.plot(x,second_deriv,'ko',label='Second Derivative', mfc='none')
plt.legend()
plt.show()

