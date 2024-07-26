#%%
import gc
import gzip
import time
import json
import shutil
import os,sys
import tldextract
import collections
import pandas as pd
from tqdm import tqdm
import urllib.request
from multiprocessing import Pool,cpu_count

CC_INDEX_PATH = 'data/cc-index.paths'
CC_CDX_EXTRACT_PATH = 'C:\\Users\\alfred\\Desktop\\D_DRIVE\\data\\cc\\'
CC_HTTP_PREFIX = 'https://data.commoncrawl.org/'
NUM_WORKERS = min(cpu_count()-1,48)

#%%
def read_every_line(fname, max_lines=-1):
    lines = []
    with open(fname, encoding='utf-8') as f:
        for i, l in enumerate(f):
            lines.append(l)
            if i>max_lines and max_lines>0:
                break
    return lines

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" % (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def save(url, filename):
    if os.path.exists(filename):
        print(f'[save] skip, already exists')
        return
    else:
        t0 = time.time()
        print(f'[save] download {url} and save to {filename}')
        urllib.request.urlretrieve(url, filename) #, reporthook)
        t1 = time.time()
        print(f'duration = {t1-t0}')

def process_index_file_line(line):
    assert type(line)==str
    
    try:
        lst = line.replace('\n','').split()
        ts = lst[1]
        data = json.loads(line.replace('\n','').split(ts)[-1].strip())
    except:
        return ()
    
    if data['status'] != '200':
        return ()
    else:
        try:
            language = data['languages']
        except:
            language = 'none'
            
        try:
            _tldextract = tldextract.extract(data['url'])
            tup = (ts,
                   data['url'],
                   _tldextract.suffix,
                   data['length'],
                   data['offset'],
                   data['filename'],
                   language              
                )
            return tup
        except:
            return ()
        
def process_index_file(file_name):
    print(f'[process_index_file] unzip index file ... ', end='')
    
    df_name = file_name.replace('.gz','.feather')
    file_unzipped = file_name.split('.gz')[0] # where to save uncompressed cdx file

    t0 = time.time()
    with gzip.open(file_name, 'rb') as f_in:
        with open(file_unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    t1 = time.time()
    print(f'duration = {t1-t0}')

    lines = read_every_line(file_unzipped,1e8)

    print('{} lines extracted'.format(len(lines)))
    
    print('Pre-processing index lines ... ')
    out = list_multiprocessing(lines, process_index_file_line, workers=NUM_WORKERS)
    
    # filter our blank lines
    out =  [_ for _ in out if _ != ()]

    print('Index pre-processed ... ')

    print('Processing index dataframe ... ')

    ts_list       = [_[0] for _ in out]
    url_list      = [_[1] for _ in out]
    tld           = [_[2] for _ in out]
    length_list   = [_[3] for _ in out]
    offset_list   = [_[4] for _ in out]
    warc_list     = [_[5] for _ in out]
    language_list = [_[6] for _ in out]

    cols = ['ts','url','tld','length','offset','warc','language']
    df = pd.DataFrame(
        data={
            'ts':ts_list,
            'url':url_list,
            'tld':tld,
            'length':length_list,
            'offset':offset_list,
            'warc':warc_list,
            'language':language_list
        },
        columns=cols
    )

    # https://en.wikipedia.org/wiki/ISO_639-3
    # zho = chinese
    # cmn = mandarin
    # yue = cantonese
    # nan = southern min
    df = df[df.language=='yue']
    df['wet'] = df.warc.apply(lambda x: x.replace('/warc/','/wet/').replace('.warc.','.warc.wet.'))
    df['wet'] = df['wet'].apply(lambda x: CC_HTTP_PREFIX + x)

    print('Index dataframe is ready ... ')
    
    os.remove(file_name)
    os.remove(file_unzipped)

    print('Files removed ... ')
    
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df.to_feather(df_name)
    
    print('Df saved ... ')

def list_multiprocessing(param_lst, func, **kwargs):
    
    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i,params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result=sorted(result,key=lambda x:x[0])
    return [_[1] for _ in result]

def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params,**kwargs)

##############################################
# PARSE CC_INDEX_PATH FILE
##############################################
#%%
cc_indexes = read_every_line(CC_INDEX_PATH,1) # only do 1 cdx file
cc_indexes = cc_indexes[:-2] # remove the meta-data / technical lines
'''
cc-index/collections/CC-MAIN-2024-22/indexes/cluster.idx
cc-index/collections/CC-MAIN-2024-22/metadata.yaml
'''
cc_indexes = [_.replace('\n','') for _ in cc_indexes] # remove line breaks

# iterate over the index files
file_dict = collections.OrderedDict()
for i,cc_index in enumerate(cc_indexes):
    if i>0:
        pass
    else:
        cc_index_file = cc_index.split('/')[-1]
        file_dict[os.path.join(CC_CDX_EXTRACT_PATH,cc_index_file)] = CC_HTTP_PREFIX + cc_index

print(len(file_dict))
#%%

##############################################
# HANDLE EACH CDX GZ FILE
##############################################
#%%
for i,(file_name,url) in enumerate(tqdm(file_dict.items())):
    print(i,file_name,url)
    print('PROCESSING INDEX FILE [{}]/[{}] ...'.format(i,len(file_dict)))
    print('Downloading an index file {} ...'.format(file_name))
    save(url, file_name)
    process_index_file(file_name)
    gc.collect()
    print(i,(file_name,url))
    print('Downloaded an index file ...')
# %%
