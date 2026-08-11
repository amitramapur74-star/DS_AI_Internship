import pandas as pd
marks=pd.Series([85,90,78],index=['math','physics','chemistry'])
print(marks['math'])
print(marks[['math','chemistry']])