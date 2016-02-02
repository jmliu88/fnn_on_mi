import os
import pdb
import numpy as np
import csv
with open('results.txt','r') as fid:
    res = fid.readlines()
    dic = []
    mod = []
    dataset = []
    acc = []
    for i in res:
        mod.append(i.split('/')[1])
        dataset.append(os.path.splitext(i.split('/')[2])[0])
    dataset = [x.split('_')[0] for x in dataset]
    mods = list(set(mod))

    pdb.set_trace()
    data = [list(set(dataset))[i] for i in [3,0,4,2,1]]
    acc =np.empty(shape=(len(mods),len(data)))
    acc.fill(np.nan)
    for i in res:
        imod = i.split('/')[1]
        idataset = os.path.splitext(i.split('/')[2])[0]
        with open(i.strip(),'r') as csvfid:
            accuracy = np.empty(shape = (1,))
            reader = csv.reader(csvfid)
            for row in reader:
                row = np.array(row).astype('float32')
                if np.all(row==0):
                   continue
                accuracy = np.append(accuracy,row)
        record = float(os.path.splitext(i.split('/')[-1])[0].split('_')[1])
        #if record > 0.5:
        acc[mods.index(imod)][data.index(idataset.split('_')[0])] =  record
#            continue
#        if idataset in ['fox_100x100_matlab','elephant_100x100_matlab','tiger_100x100_matlab']:
#            acc[mods.index(imod)][data.index(idataset.split('_')[0])] = np.mean(accuracy)/20
#        if  idataset in ['musk1norm_matlab']:
#            acc[mods.index(imod)][data.index(idataset.split('_')[0])] = np.mean(accuracy)/9.2 #float(os.path.splitext(i.split('/')[-1])[0].split('_')[-1])
#        if  idataset in ['musk2norm_matlab']:
#            acc[mods.index(imod)][data.index(idataset.split('_')[0])] = np.mean(accuracy)/10.2 #float(os.path.splitext(i.split('/')[-1])[0].split('_')[-1])

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
