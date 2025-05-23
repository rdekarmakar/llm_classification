import pandas as pd

# df = pd.read_csv('test.csv', encoding='utf-8', errors='ignore')

df = pd.read_csv('test.csv', encoding='ISO-8859-1')
print(df)