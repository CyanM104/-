import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-np.pi,np.pi,100)
y=np.sin(x)
plt.xlim(-np.pi,np.pi)
plt.xlabel(r"$\omega$") #레이텍 문법을 사용하여 수식을 입력한다.
plt.ylabel(r"$\sin (\omega)$")
plt.ylim(-1.5,1.5)
plt.plot(x,y,'ro')
plt.show()

#data=np.loadtxt("my_star.dat",float)
#x=data[:,0]
#y=data[:,1]
#plt(x,y,'ro')
#plt.show()