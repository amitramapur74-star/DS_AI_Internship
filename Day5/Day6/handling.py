import pandas as pd
data=pd.Series([10,None,30,None])
print(data.isnull())
print(data.fillna(0))