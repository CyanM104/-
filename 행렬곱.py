import numpy as np
A=np.array([[1,2,1],[0,2,3],[4,2,-1]])
B=np.array([[1,0,0],[0,1,0],[0,0,1]])
AB=np.zeros_like(A)
i=0
while i<3:
    j=0
    while j<3:
        k=0
        while k<3:
            AB[i][j]+=A[i][k]*B[k][j]
            k+=1
        j+=1
    i+=1
print("AB=")