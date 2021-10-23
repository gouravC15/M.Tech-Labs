import numpy as np
from time import *
from threading import Thread

#creating matrix


def matmultiply():
  matrix_1 = np.random.rand(10, 10)
  matrix_2 = np.random.rand(10, 10)
  result = np.random.rand(10, 10)
  for i in range(len(matrix_1)):
    for j in range(len(matrix_2[0])):
      for k in range(len(matrix_2)):
        result[i][j] += matrix_1[i][k] * matrix_2[k][j]
  for r in result:
    print(r)

start = time()
matmultiply()
end = time()
print("\nSeq Done")

#Creating obj
thread1= Thread(target=matmultiply)
start2 = time()
#Running thread
thread1.start()
end2 = time()
print("\nThread Done")

#printing exe time
sleep(0.3)
print("--------------------------------------------------------------------")
print("\nExecution time for SEQUENTIAL is: ", end - start,"\nExecution time for THREADING is: ", end2 - start2)

#Practical 2
