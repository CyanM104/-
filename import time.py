import time
res=0
num=1
infinite=int(input("Enter your calculatable number :"))
start =time.time()
while num<=infinite:
    res=num
    num+=1
    print("count ",res)
    if num > infinite:
        break
end =time.time()
print("Time taken ",end-start)
print("Total time is ",((6.02214076)*(10**23))/(end-start))
print("It takes %d years" %((10**23)/(end-start)/3600/24/365 ))