import os
import numpy as np
with open('result.txt','r') as fid:
    res = fid.readlines()
    dic = []
    mod = []
    dataset = []
    acc = []
    for i in res:
        mod.append(i.split('/')[1])
        dataset.append(os.path.splitext(i.split('/')[2])[0])
    mods = list(set(mod))
    data = [list(set(dataset))[i] for i in [3,0,4,2,1]]
    acc =np.zeros(shape=(len(mods),len(data)))
    for i in res:
        imod = i.split('/')[1]
        idataset = os.path.splitext(i.split('/')[2])[0]
        acc[mods.index(imod)][data.index(idataset)] = float(os.path.splitext(i.split('/')[-1])[0].split('_')[-1])
    print mods
    print data
    print acc
    mods_colomn = np.transpose(np.array([mods]))
    data_row = np.array([[0]+data])
    final_table = np.concatenate((mods_colomn,acc),axis=1)
    final_table = np.concatenate((data_row,final_table),axis = 0)
    print final_table
import csv
with open("results.csv",'w') as wfid:
    writer = csv.writer(wfid)
    writer.writerows(final_table)
