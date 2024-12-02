import pandas as pd

csv_filename = 'linear_database_newbl.csv'
csv_filename = 'cores_test.csv'
df = pd.read_csv(csv_filename, low_memory=False)
print(df)
avails = [(df[i], len(df[i].dropna())) for i in df]
print(avails)