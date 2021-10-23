from time import *
from threading import Thread

'''Demonstrate advantage of Scalar expansion Sequential & MultiThread'''

A = list(range(1, 901))
B=list(range(len(A)))
TS=list(range(len(A)))

def without_scalar():
	for i in A:
		T=A[i]
		A[i]=B[i]
		B[i]=T

def with_scalar():
	for i in A:
		TS[i]=A[i]
		A[i]=B[i]
		B[i]=TS[i]
	T=TS[i]

start = time()
without_scalar()
end = time()
start2 = time()
with_scalar()
end2 = time()

'''Using multi threading'''
# Creating obj
thread1 = Thread(target=without_scalar())
# Running thread and calculating time
start3 = time()
thread1.start()
end3 = time()

# Creating obj
thread2 = Thread(target=with_scalar())
# Running thread and calculating time
start4 = time()
thread2.start()
end4 = time()

print("-------Sequential----------")
print("Without Scalar expansion: ",end - start)
print("\nWith Scalar expansion: ",end2 - start2)

print("\n-----Multi Threading-----")
print("Without Scalar expansion: ",end3 - start3)
print("\nWith Scalar expansion: ",end4 - start4)


