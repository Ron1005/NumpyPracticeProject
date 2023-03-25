#Loading Numpy
import numpy as np

#Basics of Numpy
a = np.array([1,2,3])

b = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])

#Getting dimension of numpy array
print(b.ndim)

#Getting Size of numpy array

print(a.size) # size of array
print(a.itemsize) # memory size of single element in array
print(a.nbytes) # total number of bytes consumed by the array

print("-- B --")
print(b.size)
print(b.itemsize)
print(b.nbytes)


arr = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])

#----------------------------------------------------------------------------

#Getting elements in numpy array
print(arr[0][0]) # Get element at [i,j] position
print(arr[0,:,]) # Get specific row [i,:]
print(arr[:,1]) # Get speciifc column [:,j]
print(arr.view())

#Working with 3-D arrays

arr3 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(arr3.view())
print(arr3[0][1][1]) # To get specific element in 3-D array always work outside in

#-------------------------------------------------------------------------------------------

#Initializing different types of arrays

print(np.zeros((5))) # will give a matrix (1D,2D,3D...) of all zeros
print(np.ones((2,2))) # will give a matrix (1D,2D,3D...) of all ones
print(np.full((2,2),7)) # will give a matrix (1D,2D,3D...) of all specified number in this case it's 7

print(np.full_like(b,4)) # will give a matrix of size passed in first argument and filled with value passed in 2nd argument

print(np.random.rand(2,2)) # will give a matrix of size i,j,.. containing random values between 0 and 1

print(np.random.randint(3,10,size=(2,2))) # will give a matrix of size i,j,.. containing random values
                                        # between the range of first and last argument

print(np.identity(3)) # gives identity matrix of size n

#-----------------------------------------------------------------
#Task-1
#Print Following Matrix
# 1 1 1 1 1
# 1 0 0 0 1
# 1 0 9 0 1
# 1 0 0 0 1
# 1 1 1 1 1

res = np.ones((5,5))
z = np.zeros((3,3))
z[1,1] = 9
res[1:4,1:4]=z
print(res)

#---------------------------------------------------------------------

#Mathematics with numpy

m1 = np.array([1,2,3,4])
print(m1+2)  # add values to all numbers in array
print(m1-2)  # subtract values from all numbers in array
print(m1*2)  # multiply values with all numbers in array
print(m1/2)  # divide values from all numbers in array
print(m1**2) # get power of n for all values in array

# Linear Algebra

mul1 = np.ones((2,3))
mul2 = np.full((3,2),2)

print(np.matmul(mul1,mul2)) # Matrix Multiplication

square1 = np.array([[7,-4,2],[3,1,-5],[2,2,-5]])
print(round(np.linalg.det(square1))) # Finding Determinant of square matrix

# Statistics

npstats = np.array([[1,2,3],[4,5,6]])
print(npstats.sum()) # gives sum of entire matrix
print(npstats.sum(axis=0)) # gives columnwise sum for matrix
print(npstats.sum(axis=1)) # gives rowwise sum for matrix

print(npstats.min()) # gives minimum value of matrix
print(npstats.max()) # gives maximum value of matrix

#--------------------------------------------------------------------------------

# Reorganizing Arrays
shape1 = np.array([[1,2,3,4],[5,6,7,8]])

print(np.reshape(shape1,(4,2))) # Reshaping Matrix (product of number of rows and columns should be same)

#Vertical Stack

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

print(np.vstack((v1,v2))) # for vstack number of columns should be same

#horizontal stack
h1 = np.ones((2,5))
h2 = np.zeros((2,3))

print(np.hstack((h1,h2))) # for hstack number of rows should be same

#-----------------------------------------------------------------------------------------------------
#Miscellaneous

misc = np.array([1,2,3,5,10,81,73,43,6])
print(misc>7) # applying boolean conditioning on numpy array
misc1 = misc[misc>7] # filtering numpy array based on condition
print(misc1)
print(misc[[1,5,6]]) # passing a list as index in numpy array to get back list as output
print(np.all(misc>5,axis=0)) #Using any function
print(np.any(misc>5,axis=0)) #Using all function
print((misc>7) & (misc < 50)) # Using conditional operators to filter numpy arrays

#------------------------------------------------------------------------------------------------------
#Task-2

task2 = np.array([[1,2,3,4,5],
                 [6,7,8,9,10],
                 [11,12,13,14,15],
                 [16,17,18,19,20],
                 [21,22,23,24,25],
                 [26,27,28,29,30]])

# index 11,12,16,17
print(task2[2:4,0:2])

#index 2,8,14,20

print(task2[[0,1,2,3],[1,2,3,4]])

#index 4,5,24,25,29,30

print(task2[[0,4,5],3:])




