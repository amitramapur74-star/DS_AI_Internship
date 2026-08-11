import pandas as pd
scores=pd.Series([45,67,89,34,90])
passed=scores[scores>60]
print(passed)