from threading import Thread
from time import *

''''#original ()
def factorial():
  st = 1
  end = 50
  for i in range(st, end+1):
    fact = 1
    for j in range(1, i+1):
        fact = fact * j
    print(f"{i}  Factorial is {fact}")
'''

def even_factorial(st, ed):
	for i in range(st, ed + 1):
		fact = 1
		if i % 2 == 0:
			for j in range(1, i + 1):
				fact = fact * j
			print(f"{i}  Factorial is {fact}" + " :EVEN No.")

def odd_factorial(st,ed):
	for i in range(st,ed + 1):
		fact = 1
		if i % 2 == 1:
			for j in range(1, i + 1):
				fact = fact * j
			print(f"{i}  Factorial is {fact}" + " :ODD No.")


# Calling function for Sequential exe
start = time()
even_factorial(1, 100)
odd_factorial(1, 100)
end = time()
print("(Sequential Done)\n")

# creating thread object
efact_obj = Thread(target=even_factorial, args=(1, 100))  # responsible for exe even_factorial()
ofact_obj = Thread(target=odd_factorial, args=(1, 100))  # responsible for exe odd_factorial()

Mstart = time()
efact_obj.start()  # Thread Started for even
ofact_obj.start()   # Thread Started for odd
Mend = time()
print("(MultiThread Done)\n")

if __name__ == "__main__":
	sleep(0.3)
	print("\nExecution time for SEQUENTIAL is: ", end - start)
	print("\nExecution time for MULTITHREAD is: ", Mend - Mstart)
