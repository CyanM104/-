import numpy as np

data = np.loadtxt("/home/physics/다운로드/stars.txt",float)
data_x=data[:,0]
data_y=data[:,1]
print("x=")
print(data_x)
print("y=")
print(data_y)

o_file=open("my_star.dat","w")
i=0
while i<len(data_x):
    print(data_x[i],data_y[i],file=o_file)
    i+=1
o_file.close()

x=np.linspace(-np.pi,np.pi,100)
print("x=")
print(x)
y=np.sin(x)
print("y=")
print(y)

sin_res=open("sin_res.dat","w")
i=0
while i<len(x):
    print(x[i],y[i],file=sin_res)
    i+=1
sin_res.close()

gnuplot_cmd="gnuplot"
gnuplot_cmd+=" <<EOF"
gnuplot_cmd+="set term dumb 75 25"
gnuplot_cmd+="set output 'sin_res.png'"
gnuplot_cmd+="plot 'sin_res.dat' u 1:2 w lp"
gnuplot_cmd+="EOF"
print(gnuplot_cmd)