import pandas as pd

csv_filename = 'cores_test.csv'
# pd.read_csv(csv_filename, index_col=0).to_excel('data/oeis.xlsx')
df = pd.read_csv(csv_filename, index_col=0)
print(df)
ids = df.columns
print(ids)
urls = [f'https://oeis.org/{i}' for i in ids]
newcols = ['sequence ID', 'URL', 'cathegory (trivial/exists/none)', 'ground truth', 'Diofantos', 'SINDy']
outdf = pd.DataFrame(columns=newcols, index=None , data={'sequence ID': ids, 'URL': urls})
outdf.to_csv('ground_truth.csv', index=None)
print(outdf)

print(urls[0])
