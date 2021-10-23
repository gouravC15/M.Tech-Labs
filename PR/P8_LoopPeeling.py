from time import *
from threading import Thread

'''Demonstrate advantage of loop peeling
DO I = 1, N                         A(1) = A(1) + A(1)
	A(I) = A(I) + A(1)              DO I = 2, N
ENDDO                                   A(I) = A(I) + A(1)
									ENDDO
'''

A= list(range(0, 9000))

def without_peeling():
	for i in A:
		A[i] = A[i] + A[i]

def with_peeling():
	A[1] = A[1] + A[1]
	for i in range(2,len(A)):
		A[i] = A[i] + A[1]


# Time Calculation
start = time()
without_peeling()
end = time()

# Creating obj
thread1 = Thread(target=with_peeling())
start2 = time()
thread1.start()
end2 = time()

print("\n:::::::::: Exe.Time :::::::::::::::::::::::::::::::")
print("[1] Sequential Without Loop Peeling: ", end - start)
print("[2] With Threading & Loop Peeling: ", end2 - start2)