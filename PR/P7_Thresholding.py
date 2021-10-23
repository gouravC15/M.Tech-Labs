from time import *
from threading import Thread

# INDEX SET SPLITTING
'''Subdivide loop into different iteration ranges to achieve PARTIAL PARALLELIZATION (when all iterations are not involved in dependence)
	1. Threshold Analysis [Threshold= dist btwn Src & Sink]

	DO I = 1, 20
		A(100-I) = A(I) + B
	ENDDO
'''

A = list(range(0, 1000))
print("Length of A:", len(A))


def without_Thresholding():
	for i in range(0, 20):
		A[i + 20] = A[i] + 2


def with_Thresholding():
	for i in range(0, 600, 20):
		for j in range(i, i + 19):
			A[j + 20] = A[j] + 2


# Time Calculation
start = time()
without_Thresholding()
end = time()

# Creating obj
thread1 = Thread(target=with_Thresholding())
start2 = time()
thread1.start()
end2 = time()

print("\n:::::::::: Exe.Time :::::::::::::::::::::::::::::::")
print("[1] Sequential Without Thresholding: ", end - start)
print("[2] With Threading & Thresholding: ", end2 - start2)