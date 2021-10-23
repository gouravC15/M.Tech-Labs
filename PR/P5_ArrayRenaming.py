from time import *
from threading import Thread

'''Demonstrate advantage of Scalar expansion Sequential & MultiThread'''
'''DO I = 1, N                              DO I = 1, N
s1 A(I) = A(I-1) + X                      s1 A$(I) = A(I-1) + X
s2 Y(I) = A(I) + Z      renamed A(I)=>    s2 Y(I) = A$(I) + Z
s3 A(I) = B(I) + C          ''''''        s3 A(I) = B(I) + C            '''

A = list(range(0, 9000))
AS = list(range(len(A)))
Y=list(range(len(A)))
B=list(range(len(A)))

def without_ArrRename():
	for i in A:
		A[i]=A[i-1]+1
		Y[i]=A[i]+2
		A[i]=B[i]+3
	print(A)

def with_ArrRename():
	for i in AS:
		AS[i]=A[i-1]+1
		Y[i]=AS[i]+2
		A[i]=B[i]+3
	print(A)

#Time Calculation
start = time()
without_ArrRename()
end = time()


# Creating obj
thread1 = Thread(target=with_ArrRename())
# Running thread and calculating time
start2 = time()
thread1.start()
end2 = time()

print("\n:::::::::: Exe.Time ::::::::::::::::::::::::::::")
print("[1] Without Array Rename: ",end - start)
print("[2] With Threading Array Rename: ",end2 - start2)

