import numpy as np

A = [np.zeros((5,3))] * 5
B = np.zeros((3,5))

for z in A:
	B = np.dot(A,B)