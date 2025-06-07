import numpy as np

a = np.array([1, 2, 3])
b = np.array([[4, 5, 6], [6,7,8]])
print(a.ndim) #lista
print(b.ndim) #matriz
print(a.shape) #tamanho da lista
print(b.shape) #tamanho da matriz
print(a)
print(b)

draw = np.ones((5,5), dtype=int)
draw[2,2] = 9
draw[1,1:4:1] = 0
draw[3,1:4:1] = 0
draw[1:4:1,1] = 0
draw[3:4:1,1] = 0
print(draw)

##or

draw2 = np.ones((5,5))
middle = np.zeros((3,3))
middle[1,1] = 9
print(middle)
draw2[1:4,1:4] = middle
print(draw2)

####Reorganizing arrays

#vertical
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

vstacked = np.vstack([v1,v2])
print(vstacked)

#horizontal
h1 = np.ones((2,3))
h2 = np.zeros((2,1))

hstacked = np.hstack((h1,h2)) ##h2 to the right
print(hstacked)