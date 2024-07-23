import pandas as pd

dfs = []
cols = ['url','tld','wet']

for i in range(1,100):
    dfs.append(pd.read_feather('../common_crawl/data/cdx-000{}.feather'.format(str(i).zfill(2)))[cols])
for i in range(100,269):
    dfs.append(pd.read_feather('../common_crawl/data/cdx-00{}.feather'.format(str(i).zfill(3)))[cols])   

df = pd.concat(dfs)
df = df.reset_index(drop=True)

chunk_size = len(df) // 10

start = 0
end = chunk_size-1
c = 0

while end < df.shape[0]:         
    chunk = df.iloc[start:end].reset_index(drop=True)
    try:
        chunk.to_feather('index_269_{}.feather'.format(str(c)))
    except (Exception) as e:
        print (e)
        print (chunk)
        print (chunk.info())
    c+=1
    start += chunk_size
    end += chunk_size