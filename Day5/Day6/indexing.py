import pandas as pd
x = {"math":83,"Science":85,"English":80}
y=pd.Series(x)
print(y)
print(x["math"])
print(y[y>80])
print(y.loc[["math"]])