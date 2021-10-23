from time import *
from threading import Thread
'''DO I = 1, N                                  DO I = 1, N
	S1:A(I) = X(I+1) + X(I)                         S1’:X$(I) = X(I+1)
	S2:X(I+1) = B(I) + 32                           S1:A(I) = X$(I) + X(I)
	ENDDO                                           S2:X(I+1) = B(I) + 32 
	                                            ENDDO    '''

A1 = list(range(0, 9000))
A=[]
X =list(range(len(A1)))
XS = list(range(len(A1)))
B=list(range(len(A1)))

def without_NodeSplit():
	for i in A:
		A[i]=X[i+1] + X[i]
		X[i+1]=B[i] + 32
	print(X)

def with_NodeSplit():
	for i in A:
		XS[i]=X[i+1]
		A[i]=XS[i]+X[i]
		X[i+1]=B[i] + 32
	print(X)

#Time Calculation
start = time()
without_NodeSplit()
end = time()

# Creating obj
thread1 = Thread(target=with_NodeSplit())
# Running thread and calculating time
start2 = time()
thread1.start()
end2 = time()


print("\n:::::::::: Exe.Time :::::::::::::::::::::::::::::::")
print("[1] Sequential Without Node Splitting: ",end - start)
print("[2] With Threading Node Splitting: ",end2 - start2)