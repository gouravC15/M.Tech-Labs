
from time import *
from threading import Thread

'''nested Loop having dependence with Sequential & MultiThread && Loop interchange with Sequential & MultiThread'''

'''
(example)
DO I = 1, N
	DO J = 1, M
		S A(I,J+1) = A(I,J) + B
	ENDDO
ENDDO'''

print("Starting....\n")

def wLinter():                                     #wLin= without Loop Interchange
	num=1
	for i in range(1, 21):
		for j in range(1, 21):
			multiply=i*j
			iammulti=multiply * num
			print(iammulti, end=' ')
		print()

def Linter():                                      # Lin= with Loop Interchange
	num=1
	for j in range(1, 21):
		for i in range(1, 21):
			multiply = i * j
			iammulti = multiply * num
			print(iammulti, end=' ')
		print()

'''-----------------------------------wLin() without Loop Interchange'''
print("WITHOUT LOOP INTERCHANGE")
start = time()
wLinter()
end = time()
print("Sequential[✓]\n")
# Creating obj
thrad1 = Thread(target=wLinter)
start2 = time()
# Running thread
thrad1.start()
end2 = time()
sleep(0.3)
print("\nWithout Interchange Done---------------------------------------------------------------------------------------------------------\n")


'''-----------------------------------Lin() with Loop Interchange'''
print("LOOP INTERCHANGE")
start3 = time()
Linter()
end3 = time()
print("Sequential[✓]\n")
# Creating obj
thrad2 = Thread(target=Linter)
start4 = time()
# Running thread
thrad2.start()
end4 = time()
sleep(0.3)
print("\nWith Interchange Done--------------------------------------------------------------------\n")
print("[RESULTS]")
print("Loop Interchange[X] Sequential[✓]: ", end - start, "\nLoop Interchange[X] THREADING[✓] ", end2 - start2)
print("\nLoop Interchange[✓] Sequential[✓]: ", end3 - start3, "\nLoop Interchange[✓] THREADING[✓] ", end4 - start4)