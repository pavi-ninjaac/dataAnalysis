# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:54:49 2020

@author: ninjaac
"""


import numpy as np
arr=np.arange(100000)
list=[x for x in range(1,100000)]

#creation odf numpy array
a=np.arange(1,10000,2)
b=np.ones(2)
b2=np.ones_like(a)
c=np.zeros((2,3),order='C')

#_like functio create the array with the same shape and size dtype to the given array
c2=np.zeros_like(b)

d=np.empty((2,3),dtype='int8',order='C')
d2=np.empty_like(c)
e=np.full((2,3),dtype='int8',order='C',fill_value=68)
e2=np.full_like(d,fill_value=10,dtype='float64')
f=np.eye(3)
 
# manupulation of the array
#not like list ans set dict 
# it cane iteratable
# easy to manupulate

#mathamtics +_*/

aa=a+a
bb=b-b

# 10 will multiply through the all elememntd in the array
cc=c*10         
o1=np.array([2,5,3])
oo2=o1/2

o=np.array([2,5,3])
oo=o//2

arr=np.array([[1,2,3],[0,6,7]])
arr1=np.array([[0,4,6],[0,2,8]])
arr>arr1

"""array([[ True, False, False],
       [False,  True, False]])"""

#indexing and slicing 

arr=np.array([
                [1,2,3],
                [0,6,7]
            ])
arr[0][2] #3

#slicing
#2d slicing
arr[:,0] #array([1, 0])

arr[::-1]
#reversing the rows
"""array([[0, 6, 7],
       [1, 2, 3]])"""

# if we changed the sliced array values it will affect the original array
# to avoid that use copy() function 
arr2=arr[:,1:3]
"""array([[2, 3],
       [6, 7]])"""
arr2[0][1]=10
arr2
"""array([[ 2, 10],
       [ 6,  7]])"""
arr 
"""array([[ 1,  2, 10],
       [ 0,  6,  7]])"""

arr2=arr[:,1:3].copy()
"""array([[2, 3],
       [6, 7]])"""
arr2[0][1]=10
arr2
"""array([[ 2, 10],
       [ 6,  7]])"""

# but when we use copy() function it will not affect the original array
arr
"""array([[1, 2, 3],
       [0, 6, 7]])"""

# transpose and swapping 

arr3=np.arange(16).reshape(2,2,4)

"""array([[[ 0,  1,  2,  3],[ 4,  5,  6,  7]],

       [[ 8,  9, 10, 11],[12, 13, 14, 15]]])"""
arr3.T
"""array([[[ 0,  8],
        [ 4, 12]],

       [[ 1,  9],
        [ 5, 13]],

       [[ 2, 10],
        [ 6, 14]],

       [[ 3, 11],
        [ 7, 15]]])"""
#matrix multiplication

np.dot(arr3.T,arr3)

#ufunctions 
# sqrt,square
#that done element wise manupulation

x1=np.random.randint(19)
#6
x=np.random.randn(19)
"""#array([-1.92230889,  1.6624133 ,  0.61117225,  0.10693814,  0.30178702,
        0.24698495, -0.16696016,  0.00447291,  0.64343305,  0.36036831,
       -0.11413755, -0.11163691, -0.84330476, -0.42197341,  1.65955176,
       -1.24222189, -0.02472928, -0.49590682,  0.20760994])"""
np.sqrt(x)
y=np.random.randint(19)
#5
y1=np.random.randn(19)
np.maximum(x,y1)



"""array([ 1.11825757, -0.22316475,  1.13377668, -0.11434464,  0.71433655,
       -0.72968478, -0.29191779, -1.47936268,  1.08198502,  1.78893011,
       -0.88464506, -0.12117126,  0.23049042,  0.98027164,  2.05552856,
       -0.71638242,  0.9222068 , -0.43455824,  0.49748931])"""

# teh meshgrid(arr,arr) used to take 2 1D array and create an 2D matrixs

arr4=np.arange(6)
np.meshgrid(arr4,arr4)
"""
[array([[0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5]]),
 array([[0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5]])]
"""

arr4=np.arange(6)
arr5=np.arange(4,8)
np.meshgrid(arr4,arr5)

"""
[array([[0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5]]),
 array([[4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7, 7]])]
"""

# use of where function
# helping in statistic when i want to fill all +ve values with 2
# and all -ve values with -2 by simply

arr6=np.random.randn(10)
"""
array([ 0.11545093, -1.48107312, -0.03672199,  1.50740325,  1.61044337,
       -0.4645235 , -1.51658968,  1.30939463,  0.15914635,  1.68024114])"""
arr6_bool=arr6>0
"""array([ True, False, False,  True,  True, False, False,  True,  True,
        True])"""

np.where(arr6>0,1,-1)
# array([ 1, -1, -1,  1,  1, -1, -1,  1,  1,  1])


#mathamatical and statistics
arr6.sum()
arr6.mean()
arr6.std(axis=0)

#mathods of boolean array
(arr6>0).sum() # 6 pasitive numbers

arr6_bool.any() # true any true value
arr6_bool.all()# flash is all values is true then true

# sorting the array
arr6.sort()# sort the values
arr6.sort(1)#sort based on axis 1 for row and 0 for column








