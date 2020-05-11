# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:59:06 2020

@author: ninjaac
"""

# 1)Import the numpy package under the name np
import numpy as np

#2)Print the numpy version and the configuration
np.__version__  #Out[51]: '1.18.1'
np.show_config() #will show our numpy configuration

#3)Create a null vector of size 10
empty_arr=np.zeros(10)    #array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


#4)How to find the memory size of any array
# memory size = # byte the array consumed
print(f"number of bytes is consumed {empty_arr.nbytes}")
# ANS:number of bytes is consumed 80

#5)How to get the documentation of the numpy add function from the command line?

%run `python -c "import numpy; numpy.info(numpy.add)"



#6)Create a null vector of size 10 but the fifth value which is 1
arr6=np.zeros(10)
arr6[4]=1 #array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

#7)Create a vector with values ranging from 10 to 49
arr7=np.arange(10,50)
"""
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
       44, 45, 46, 47, 48, 49])"""

#8)Reverse a vector
arr8=np.arange(1,10)[::-1]  # array([9, 8, 7, 6, 5, 4, 3, 2, 1])

#9)Create a 3x3 matrix with values ranging from 0 to 8
arr9=np.arange(0,9).reshape(3,3)
"""
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])"""

#10)Find indices of non-zero elements from [1,2,0,0,4,0]
arr10=np.nonzero([1,2,0,0,4,0]) #(array([0, 1, 4], dtype=int32),)

#11)Create a 3x3 identity matrix 
arr11=np.eye(3)
"""
Out[73]: 
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])"""

#12)Create a 3x3x3 array with random values
arr12=np.random.randn(27).reshape(3,3,3)
"""
array([[[-1.14649639,  0.21246582, -1.5388296 ],
        [ 0.14170138, -1.31096577, -0.65416288],
        [ 1.30018706,  0.32283504,  1.39123551]],

       [[ 0.9642615 ,  1.03453637, -0.03462993],
        [-1.55945517, -0.36986315,  0.98714535],
        [ 1.292804  ,  0.62701675, -2.35813021]],

       [[ 0.28899303, -0.53329878,  0.42575985],
        [-2.02820402, -0.1355543 ,  1.01161945],
        [-0.99083261,  3.74670899,  1.52951764]]])"""

#13)Create a 10x10 array with random values and find the minimum and maximum values 
arr13=np.random.randn(10,10)
#column wish minimum  element
col=arr13.min(axis=0)
row=arr13.min(axis=1)
#col wish max element
col_max=arr13.max(axis=0)
row_max=arr13.max(axis=1)
#total min and max element
min_,max_=arr13.min(),arr13.max() #Out[84]: -2.5712329156945923

#14)Create a random vector of size 30 and find the mean value
a14=np.random.randn(30).mean()  #Out[86]: -0.20202871269060171

#15)Create a 2d array with 1 on the border and 0 inside
a15=np.zeros((5,5),dtype='int64')
a15[:,(0,4)]=1
a15[0]=1
a15[4]=1


#16)How to add a border (filled with 0's) around an existing array?
a16 = np.ones((5,5))
a16= np.pad(a16, pad_width=1, mode='constant', constant_values=0)
"""
[[0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 1. 1. 1. 1. 0.]
 [0. 1. 1. 1. 1. 1. 0.]
 [0. 1. 1. 1. 1. 1. 0.]
 [0. 1. 1. 1. 1. 1. 0.]
 [0. 1. 1. 1. 1. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]"""

#17)What is the result of the following expression?
print(0 * np.nan) #nan nan * anything is nan
print(np.nan == np.nan) #False
print(np.inf > np.nan) #False
print(np.nan - np.nan) #nan any arithmatica operation will give you nan
print(np.nan in set([np.nan])) #True
print(0.3 == 3 * 0.1) #Flase because 3*0.1 will give 0.300000000000


#18)Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
#diag will return the diagonal k=1 upper diag k=-1 lower diag
#a18=np.diag() will give a empty aray first function will give the values to be filed

a18=np.diag(np.arange(1,5),k=-1)
"""
Out[120]: 
array([[0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 2, 0, 0, 0],
       [0, 0, 3, 0, 0],
       [0, 0, 0, 4, 0]])"""

#19) Create a 8x8 matrix and fill it with a checkerboard pattern
a19=np.zeros((8,8))
a19[1::2,0::2]=1
a19[0::2,1::2]=1
"""
array([[0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.],
       [0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.],
       [0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.],
       [0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.]])
"""
#20) Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element
print(np.unravel_index(99,(6,7,8)))

#21)Create a checkerboard 8x8 matrix using the tile function
#tile(a,reps) repeat a reps times
a21=np.tile(np.array([[1,0],
                     [0,1]]),(4,4))


#22) Normalize a 5x5 random matrix
#[-1,1]
a22=np.random.randn(25).reshape(5,5)
a22_nor=(a22-np.mean(a22))/np.max(a22)
"""
array([[ 0.10461079,  0.17451335, -0.3982498 ,  0.04206602, -0.21858879],
       [-0.23829601,  0.2180956 ,  0.55588981, -0.30542219,  0.24784357],
       [ 0.35715276,  0.67892603,  0.57485102,  1.06076041, -0.70792837],
       [-0.31586796,  0.17991482, -0.44580749, -0.49352988, -0.53920214],
       [-0.59282106,  0.06737855, -0.80045191,  0.04617754,  0.74798533]])"""

#23)Create a custom dtype that describes a color as four unsigned bytes (RGBA)
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])

#24)Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
a24=np.dot(np.arange(15).reshape(5,3),np.arange(6).reshape(3,2))
"""
Out[21]: 
array([[ 10,  13],
       [ 28,  40],
       [ 46,  67],
       [ 64,  94],
       [ 82, 121]])"""

#25 Given a 1D array, negate all elements which are between 3 and 8, in place)
a25=np.arange(12)
a25[(a25>3) & (a25<8) ] *=-1
#Out[25]: array([ 0,  1,  2,  3, -4, -5, -6, -7,  8,  9, 10, 11])

#26)What is the output of the following script?
print(sum(range(5),-1)) #10
from numpy import *
print(sum(range(5),-1)) #10

#27)Consider an integer vector Z, which of these expressions are legal?
"""
Z**Z #array([   1,    1,    4,   27,  256, 3125], dtype=int32) legal
2 << Z >> 2 array([ 0,  1,  2,  4,  8, 16], dtype=int32) legal
Z <- Z   array([False, False, False, False, False, False]) legal
1j*Z  array([0.+0.j, 0.+1.j, 0.+2.j, 0.+3.j, 0.+4.j, 0.+5.j]) legal
Z/1/1n   array([0., 1., 2., 3., 4., 5.])
Z<Z>Z    notlegal                             """
z=np.arange(6)


#28)What are the result of the following expressions
np.array(0) / np.array(0) #0
np.array(0) // np.array(0) #0
np.array([np.nan]).astype(int).astype(float) #array([-2.14748365e+09])

#29)How to round away from zero a float array 
Z = np.random.uniform(-10,+10,10)
 
print (np.copysign(np.ceil(np.abs(Z)), Z))



#30)How to find common values between two arrays?
a30 = np.random.randint(0,10,10)
a302 = np.random.randint(0,10,10)
print(np.intersect1d(a30,a302))

#31)How to ignore all numpy warnings (not recommended)?
default=np.seterr(all="ignore")
np.ones(1) / 0


#32)Is the following expressions true?
np.sqrt(-1) == np.emath.sqrt(-1) # false

#33)How to get the dates of yesterday, today and tomorrow
today=np.datetime64('today')
yesterday=np.datetime64('today')-np.timedelta64(1) #numpy.datetime64('2020-05-09')
tomo=np.datetime64('today')+np.timedelta64(1) #numpy.datetime64('2020-05-11')



#34)How to get all the dates corresponding to the month of July 2016?
a34=np.arange('2016-07','2016-08',dtype="datetime64[D]")
"""array(['2016-07-01', '2016-07-02', '2016-07-03', '2016-07-04',
       '2016-07-05', '2016-07-06', '2016-07-07', '2016-07-08',
       '2016-07-09', '2016-07-10', '2016-07-11', '2016-07-12',
       '2016-07-13', '2016-07-14', '2016-07-15', '2016-07-16',
       '2016-07-17', '2016-07-18', '2016-07-19', '2016-07-20',
       '2016-07-21', '2016-07-22', '2016-07-23', '2016-07-24',
       '2016-07-25', '2016-07-26', '2016-07-27', '2016-07-28',
       '2016-07-29', '2016-07-30', '2016-07-31'], dtype='datetime64[D]')"""

#35)How to compute ((A+B)*(-A/2)) in place (without copy)?
A=np.arange(5)
B=np.arange(7,12)
c=A+B
d=(-A )/2  
an=c*d
#using functions
"""
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(B,A)"""
#36)Extract the integer part of a random array using 5 different methods
a36=np.random.uniform(0,10,10)
a36.astype(int) # array([6, 6, 5, 5, 2, 5, 0, 5, 9, 5])
np.floor(a36)
np.ceil(a36)-1

#37)Create a 5x5 matrix with row values ranging from 0 to 4
np.tile(np.arange(5),(5,1))
"""
array([[0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4]])"""
#38)Consider a generator function that generates 10 integers and use it to build an array
def gen():
    for i in range(10):
        yield i
x=[g for g in gen()]        
np.array(x)
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#39)Create a vector of size 10 with values ranging from 0 to 1, both excluded
np.linspace(0,1,11,endpoint=False)[1:]
"""
Out[119]: 
array([0.09090909, 0.18181818, 0.27272727, 0.36363636, 0.45454545,
       0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091])"""
#40)Create a random vector of size 10 and sort it 
a40=np.random.randn(10)
a40.sort()

#41)How to sum a small array faster than np.sum
a41=np.arange(100000)
np.add.reduce(a41)

#42)Consider two random array A and B, check if they are equal 
A=np.random.random(12)
B=np.random.random(12)

np.array_equal(A,B) #False
#anothe method
A=np.array([1,2])
B=np.array([1,2])
(A==B).all()

#43) Make an array immutable
a43=np.arange(10)
a42.flags.writable=False

#44)Consider a random 10x2 matrix representing cartesian coordinates, convert them to 
#polar coordinates 
#polor (r,0)
a44=np.random.randint(0,10,(10,2))
x,y=a44[:,0],a44[:,1]
r=np.sqrt(x**2+y**2)
teta=np.arctan2(y,x)
#polor to cartisian
polor=[]
for i,a in zip(r,teta):
    polor.append([i,a])
a=np.array(polor)
r,teta=a[:,0],a[:,1]
x=r*np.cos(teta)
y=r*np.sin(teta)

#45)Create random vector of size 10 and replace the maximum value by 0
a45=np.random.rand(10)
a45[a45.argmax()]=0

#46)Create a structured array with x and y coordinates covering the [0,1]x[0,1] area
Z = np.zeros((5,5), [('x',float),('y',float)])
#will give you the matrix from the vector
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)

#47)Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
x=np.random.randint(1,11,(3,2))
y=np.random.randint(1,11,(3,2))
c=1/(np.subtract.outer(x,y))

#48)Print the minimum and maximum representable value for each numpy scalar type
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)

#49How to print all the values of an array)
print(np.full((10,10),fill_value=10))

#50) How to find the closest value (to a given scalar) in a vector
x=np.random.randint(1,11,10)
choice=np.random.randint(1,5)
x[np.abs(x-choice).argmin()]

#51)Create a structured array representing a position (x,y) and a color (r,g,b)
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)

#52) Consider a random vector with shape (100,2) representing coordinates, find point by point distances
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)


#53)How to convert a float (32 bits) array into an integer (32 bits) in place
a53=np.arange(13,dtype='float32')
a53.astype('int32') #will give you the ineger part
#Out[47]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
#return without float
a53.view('int32')
"""
array([         0, 1065353216, 1073741824, 1077936128, 1082130432,
       1084227584, 1086324736, 1088421888, 1090519040, 1091567616,
       1092616192, 1093664768, 1094713344])"""

#54)How to read the following file

"""1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11"""
from io import StringIO #act like a file object
c=StringIO('''
           1, 2, 3, 4, 5
           6,  ,  , 7, 8
           ,  , 9,10,11
           ''')
#loadtxt will fater than gen for smaller files           
np.loadtxt(c,dtype='str',delimiter=',')
#gen will handle missing values too
np.genfromtxt(c,delimiter=',')

#55)equalent function equal to enumerate
a55=np.arange(8)
for i,value in np.ndenumerate(a55):
    print(i,value)

#56)Generate a generic 2D Gaussian-like array
"""X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)    
"""    
    
#57)How to randomly place p elements in a 2D array
#use choice method
p=3
a=np.arange(25).reshape(5,5)
for i in range(5):
    print(np.random.choice(a.ravel(),size=p,replace=True)) #Out[50]: array([ 0, 10, 23])
"""
[1 5 0]
[ 8 11 24]
[17 21  8]
[ 1 12  0]
[ 0 18 18]"""

#58)Subtract the mean of each row of a matrix        
a58=np.arange(12).reshape(2,6)
a=a58-a58.mean(axis=1,keepdims=True)

#59) How to sort an array by the nth column
a59=np.random.randint(1,15,(3,5))
col=3
a59[a59[:,col].argsort()]

#60)How to tell if a given 2D array has null columns
a6=np.array(
    [[1,2,3,4,np.nan],
     [2,3,4,5,np.nan]])
np.where(np.isnan(a6).any(axis=0))
# (array([4], dtype=int32),)

#61) Find the nearest value from a given value in an array
a61=np.arange(12).reshape(3,4)
a61=a61.ravel()
a=4.56
i=a61[np.abs(a61-a).argmin()]
a61[i] #5

#62)Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
A=np.arange(3).reshape(1,3)
B=np.arange(3).reshape(3,1)
#using both give same output
np.add(A,B)
it = np.nditer([A,B,None])
for x,y,z in it: z = x + y
print(it.operands[2])

#63)Create an array class that has a name attribute
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name) 
#64)Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)
a64=np.arange(6)
index=np.array([2,3])
for i in index:
    a64[i]=a64[i]+1
#Out[104]: array([0, 1, 3, 4, 4, 5])


#65)How to accumulate elements of a vector (X) to an array (F) based on an index list (I)
x=np.arange(1,7)
i=np.array([0,3,4,2,5,1])
f=np.bincount(i,x).astype('int64')


#66) Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
'''w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))'''

#67)Considering a four dimensions array, how to get sum over the last two axis at once
a67=np.random.randint(0,11,(2,1,2,3))
sum = a67.sum(axis=(-2,-1))



#68)Considering a one-dimensional vector D, how to compute means of subsets 
#of D using a vector S of same size describing subset indices
D=np.array([1,2,3,4,5,6])
S=np.array([0,2,4])
sum=0
for i in S:
    sum+=D[i]
mean=sum/len(S)
print(mean)

#69)How to get the diagonal of a dot product
a=np.random.randint(1,11,(5,5))
b=np.random.randint(1,11,(5,5))
np.diag(np.dot(a,b))

#70) Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? 
a70=np.array([1,2,3,4,5])
a1=[]
for i in a70:
    a1.extend([i,0,0,0])
    
#71) Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)
a71=np.arange(75).reshape((5,5,3))
a=np.arange(25).reshape((5,5))
ans=a71*a[:,:]

#72) How to swap two rows of an array
b=np.arange(6).reshape(2,3)
b[(0,1),:]=b[(1,0),:]
#b[[0,1]]= b[[1,0]]
print(b)

#73)Consider a set of 10 triplets describing 10 triangles (with shared vertices), 
#find the set of unique line segments composing all the triangles
faces = np.random.randint(0,100,(10,3))
f = np.roll(faces.repeat(2,axis=1),-1,axis=1)
f = F.reshape(len(f)*3,2)
f = np.sort(f,axis=1)
g = F.view( dtype=[('p0',f.dtype),('p1',f.dtype)] )
g = np.unique(g)


#74)Given an array C that is a bincount, how to produce an array A such that 
#np.bincount(A) == C?

a=np.array([1,2,3,5,3,4,5,6,2,4])
c=np.bincount(a)
np.repeat(np.arange(len(c)),c)
"""
np.repeat(np.array([[1,2],[2,3]]),[2,3] ,axis=0)
np.repeat(np.array([[1,2],[2,3]]),2,axis=1)
np.repeat(x,2)
"""
#75)How to compute averages using a sliding window over an array
a=np.array([1,2,3,5,3,4,5,6,2,4])

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))

#76)Consider a one-dimensional array Z, build a two-dimensional array whose first row 
#is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 
#(last row should be (Z[-3],Z[-2],Z[-1])
from numpy.lib import stride_tricks
def shift(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = shift(np.arange(10), 3)
print(Z)

#stride_tricks.as_strided(np.arange(6), shape=(3,2), strides=(8,8))

#77) How to negate a boolean, or to change the sign of a float inplace
a77=np.random.randn(10).reshape(5,2)
np.negative(a77) #plu go to minus vise versa

np.logical_not([True,False,0,1]) # array([False,  True,  True, False])

#78) Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to
# compute distance from p to each line i (P0[i],P1[i])
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
#79)Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))



#80)Consider an arbitrary array, write a function that extract a subpart with 
#a fixed shape and centered on a given element (pad with a fill value when necessary)
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

#81)Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
# how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]
from numpy.lib import stride_tricks
Z =np.array( [1,2,3,4,5,6,7,8,9,10,11,12,13,14])
stride_tricks.as_strided(Z, shape=(11,4), strides=(4,4))
#Z.size=14 so 14-4+1=11  4 for 4 columns

#82) Compute a matrix rank
from numpy.linalg import matrix_rank
a82=np.arange(30).reshape(5,6)
matrix_rank(a82)
# 2

#83) How to find the most frequent value in an array
a83=np.array([1,2,3,2,4,2,4,5,6,7,6,6,7,8,6])
np.bincount(a83).argmax()   #6
times=np.bincount(a83).max() #4 times

#84)Extract all the contiguous 3x3 blocks from a random 10x10 matrix
#np.arange(54).reshape(2,3,3,3)
from numpy.lib import stride_tricks
a84= np.random.randint(0,5,(10,10))
stride_tricks.as_strided(Z, shape=(8,8,3,3), strides=a84.strides + a84.strides)

#10-3+1=8(shape[0] 10-3+1=8shape[1] for 10 in shape(a84)=(10,10)

#85)Create a 2D array subclass such that Z[i,j] == Z[j,i]
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)

#86)
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
                                      
#87)Consider a 16x16 array, how to get the block-sum (block size is 4x4)
Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
#88)How to implement the Game of Life using numpy arrays
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
                    
#89)find n largest values in array
a89=np.arange(10000)
n=6
a89.argsort()[-n:]
# array([9994, 9995, 9996, 9997, 9998, 9999], dtype=int32)

#90) Given an arbitrary number of vectors, build the cartesian product (every combinations of every item)

def cartesian(arrays):
    arrays=[np.asarray(x) for x in arrays]
    shape=[len(x) for x in arrays]
    len=1
    for i in range(0,len(arrays)):
        len*=len(arrays[i])
    indexices_x=np.indices(len,dtype='int64')
    for n, arr in enumerate(arrays):
        indexices_x[:, n] = arrays[n][indexices_x[:, n]]
    return indexices_x

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))   
    
    
    
#91) How to create a record array from a regular array
a91=np.array([[1,2],[34,56],['pavi','ninja']])
np.core.records.fromarrays(a91,names=['ex_no','age','name'])
"""
rec.array([('1', '34', 'pavi'), ('2', '56', 'ninja')],
          dtype=[('ex_no', '<U11'), ('age', '<U11'), ('name', '<U11')])"""

#92)Consider a large vector Z, compute Z to the power of 3 using 3 different methods
a92=np.random.randint(10)
np.power(a92,3)
a92*a92*a92
#729
#93)Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B


#94)Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])

a94 = np.array([[2, 4, 2],
       [1, 1, 3],
       [1, 4, 0],
       [3, 2, 2],
       [0, 2, 2],
       [0, 1, 1],
       [4, 0, 1],
       [1, 1, 1],
       [1, 2, 3],
       [2, 3, 3]])     
U=a94[[(np.bincount(a94[i]).max())!=len(a94[i]) for i in range(0,len(a94))],:]

#another logic   U = Z[Z.max(axis=1) != Z.min(axis=1),:]
#U = a94[(np.bincount(a94[i]).max())!=len(a94[i]),:]
#95)

#96)Given a two dimensional array, how to extract unique rows
a95=np.random.randint(1,10,(5,5))
np.unique(Z, axis=0)


#97)Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function 
A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)             
np.einsum('i,j->ij', A, B)    # np.outer(A, B)

#98)Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y).8i


#99) Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])


#100) Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)













