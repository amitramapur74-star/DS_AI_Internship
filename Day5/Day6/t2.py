import pandas as pd
s1 = pd.Series([10,20,30,40])
s2 = pd.Series([10,20,30],index=['a','b','c'])
print(s1)
print(s2)
print(s2.tolist())