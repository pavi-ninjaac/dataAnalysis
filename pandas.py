# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:43:49 2020

@author: ninjaac
"""
#purely pandas

import pandas as pd
#Series
pd0=pd.Series([1,2,3,4,5,6])
"""0    1
1    2
2    3
3    4
4    5
5    6
dtype: int64"""
pd1=pd.Series([1,2,3,4,5,6],index=[5,6,7,8,9,0])
"""5    1
6    2
7    3
8    4
9    5
0    6
dtype: int64"""
#pandas using dict

a={"a":1,"b":2,"c":3}
pd2=pd.Series(a)
#automatically keys are gonna to be the index and values are in values
"""Out[30]: 
a    1
b    2
c    3
dtype: int64"""
#index and valules function

pd2.index
pd2.values#array([1, 2, 3], dtype=int64)

pd2.name="alexnev"

pd2
"""a    1
b    2
c    3
Name: alexnev, dtype: int64"""
pd2.index.name="indaydg"

#access through keys that is index

pd2['b']=34
"""a     1
b    34
c     3
Name: alexnev, dtype: int64"""
pd2.isnull()
"""a    False
b    False
c    False
Name: alexnev, dtype: bool"""
#addition ,-,*,/ aplied for each entities

pd2+2
"""a     3
b    36
c     5
Name: alexnev, dtype: int64"""

pd3=pd.Series(a,index=['a','b','c','d'])
#no d so put NaN(not a number )
"""a    1.0
b    2.0
c    3.0
d    NaN
dtype: float64"""
#addition between series
pd2+pd3
"""a     2.0
b    36.0
c     6.0
d     NaN
dtype: float64"""
#dataframes

dict={"year":[1001,1002,1003,1004],
      "age":[34,56,45,32],
      "name":['pavi','ac','ninja','kd'],}
df=pd.DataFrame(dict)


"""   year  age   name
0  1001   34   pavi
1  1002   56     ac
2  1003   45  ninja
3  1004   32     kd"""
#df.head() will throw the first 5 rows
df['year']
df['sex']=(df['age']>0)
"""   year  age   name   sex
0  1001   34   pavi  True
1  1002   56     ac  True
2  1003   45  ninja  True
3  1004   32     kd  True"""

del df['sex']
"""   year  age   name
0  1001   34   pavi
1  1002   56     ac
2  1003   45  ninja
3  1004   32     kd"""
df.columns





















