import dask.array as da
import numpy as np
import dask.bag as db
import pandas as pd
import dask.dataframe as dd
import os
import shutil
path =  'D:\\Predictionofsales\\needingdata\\anli\\'
openFilename = 'qq_20190226155610.csv'
save_path =  'D:\\Predictionofsales\\needingdata\\anli\\anli_fdc1\\'
filename = '-*.csv'
df = dd.read_csv(path + openFilename, dtype=str, encoding='utf-8', engine='python', sep=',')
jd_wh_store_no = 'jd_wh_store_no'
stores = df[jd_wh_store_no].unique().compute().tolist()
print(stores[:5])
storelist = []
for store in stores:
    a = df[df[jd_wh_store_no] == store]
    dd.to_csv(a, filename=save_path+filename, index=False, sep=',')
    filelist = os.listdir(save_path)
    # print(filelist)
    if len(storelist) != 0:
        for i in storelist:
            filelist.remove(i)
    combined_csv = pd.concat( [ pd.read_csv(save_path+f) for f in filelist])
    combined_csv.to_csv(save_path+str(store)+".csv", index=False)
    storelist.append(str(store)+".csv")

    for f in filelist:
        filepath = os.path.join(save_path, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
