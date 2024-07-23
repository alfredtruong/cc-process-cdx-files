import gc
import time
import pickle 
import pandas as pd

def pckl(obj,path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def upkl(path):
    with open(path, 'rb') as handle:
        _ = pickle.load(handle)
    return _

dfs = []
# get whole index files
for i in range(0,10):
    dfs.append(pd.read_feather('index_269_{}.feather'.format(str(i))))
    print('Index {} loaded'.format(i))
df = pd.concat(dfs)
del dfs

gc.collect()
print('Memory released')
time.sleep(10)

# rank wet files by their popularity within Russian websites
wet_urls = list(df.wet.value_counts().index)
url_set = set(df.url.unique())

pckl(wet_urls,'wet_urls.pickle')
pckl(url_set,'url_set.pickle')