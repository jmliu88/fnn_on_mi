import os
import sys
import numpy as np
import htkmfc
import h5py
N_FEATURE_DIM = 39
MAX_LENGTH = 5000
label_index = ['KX','FN','XQ','JJ','HP','HQ','sad','noise']
def readHTKfeat(dataPath):
    data = np.array([[]])
    flagDataStart = 0
    nLine = 0
    i = -1
    for line in open(dataPath,'r'):
        if line.find('END') != -1:
            return data
        if flagDataStart:
            if nLine == 0:
                dataLine = np.array([float(x) for x in line.split()[1:]])
                nLine = nLine + 1
                i = i+1
                continue
            elif nLine in [1,2]:
                dataLine = np.concatenate((dataLine,[float(x) for x in line.split()]))
                nLine = nLine + 1
                continue
            elif nLine == 3:
                dataLine = np.concatenate((dataLine,[float(x) for x in line.split()]))
                data[i,:] = dataLine
                nLine = 0
                continue

        if line.find('------------------------------------ Samples: 0->-1 ------------------------------------') != -1:
            flagDataStart = 1

        if line.find('Num Samples') != -1:
            nSamples = int(line.split()[2])
            data = np.ndarray(shape=(nSamples,N_FEATURE_DIM))
            flagDataStart= 0

    return data

def read_from_list(f_list):
    for iFile in f_list:
        f = h5py.File(iFile.strip())
        if 'X' not in locals():
            X = f['data'][:,2000:7000:1,:]
            Y = f['label'][:]
            mask = f['mask'][:,2000:7000:1]
        else:
            X = np.concatenate((X,f['data'][:,2000:7000:1,:]),axis=0).astype('float32')
            Y = np.concatenate((Y,f['label'][:]),axis=0).astype('float32')
            mask = np.concatenate((mask,f['mask'][:,2000:7000:1]),axis=0).astype('float32')
        f.close()
    return (X,Y,mask)

def read_htk(f,length = 5000):
    fid = htkmfc.open(f.strip())
    mfc = fid.getall()
    fid.close()
    if mfc.shape[0] < length:
        return np.concatenate((mfc,np.zeros([length-mfc.shape[0],mfc.shape[1]])),axis = 0), np.concatenate((np.ones([mfc.shape[0],1]),np.zeros([length-mfc.shape[0],1])),axis = 0)

    else:
        return (mfc[:length,:],np.ones([length,1]))

def name_to_label(filename):
    for i,l in enumerate( label_index):
        if filename.find(l) != -1:
            return i
def label_to_name(lab):

    return label_index[lab]
def read_htk_list(f_list):
    for iFile in f_list:
        #f = htkmfc.open(iFile.strip())
        x,m = read_htk(iFile)
        if 'X' not in locals():
            X = np.expend_dims(x,axis=0)
            Y = np.array([name_to_label(iFile)])
            mask = np.expend_dims(m,axis=0)
        else:
            X = np.concatenate((X,np.expend_dims(x,axis=0)),axis=0).astype('float32')
            Y = np.concatenate((Y,np.array([name_to_label(iFile)])),axis=0).astype('int32')
            mask = np.concatenate((mask,np.expend_dims(m,axis=0)),axis=0).astype('float32')
    return (X,Y,mask)
def format_htktxt_to_h5():
    ## depreciated
    h5File = 'mfcc.h5'
    if os.path.exists(h5File):
        print('loading from h5 file...')
        sys.stdout.write("\033[F")
        f = h5py.File(h5File,'r')
        X = f['X'][:]
        Y = f['Y'][:]
        mask = f['mask'][:]
        N_SEQ = len(Y)
        MAX_LENGTH=15000
        f.close()
        print('DONE')
    else:
        ## Load Feature
        dataDir = '/home/james/data/MFCC/MFCC_E_D_A_txt/'
        featDic = {}
        dirList=os.listdir(dataDir)
        for i in range(len(dirList)):
            if i%100 == 0:
                print('reading %d,'%i)
            dataPath = os.path.join(dataDir,dirList[i])
            try:
                data = readHTKfeat(dataPath)
                featDic.update({dirList[i]:data})
            except:
                print(dataPath)
        print('Loading DONE')
        ## Read label
        lab = ['angry','sad','inneed','happy','alert','scary','nobark']
        labDic = {}
        for k in featDic:
            for i,l in enumerate(lab):
                if k.find(l) != -1:
                    labDic[k] = i
                    continue
        ## Format Data
        # Number of sequences
        N_SEQ = len(labDic)
        # Max sequence length
        MAX_LENGTH = 15000
        ## Format data into X Y and mask
        X = np.zeros(shape=(N_SEQ, MAX_LENGTH, N_FEATURE_DIM))
        mask = np.zeros((N_SEQ, MAX_LENGTH))
        Y = np.zeros((N_SEQ,))
        for i,k in enumerate(labDic):
            X[i,0:len(featDic[k]),:] = featDic[k]
            mask[i] = [1]*len(featDic[k])+[0]*(MAX_LENGTH-len(featDic[k]))
            Y[i] = labDic[k]
        Y = Y.astype('int32')
        del featDic
        del labDic
        print('Formating data DONE')
        ## Save Data
        f = h5py.File('mfcc.h5','w')
        f.create_dataset('X',data=X)
        f.create_dataset('Y',data=Y)
        f.create_dataset('mask',data=mask)
        f.close()

def format_data_to_rm(dataDir,saveDir):
    ## currentlly useless
    featDic = {}
    dirList=os.listdir(dataDir)
    for i in range(len(dirList)):
        if i%100 == 0:
            print('reading %d,'%i)
        dataPath = os.path.join(dataDir,dirList[i])
        try:
            data = readHTKfeat(dataPath)
            featDic.update({dirList[i]:data})
        except:
            print(dataPath)
            continue
        ## Read label
        lab = ['angry','sad','inneed','happy','alert','scary','nobark']
        for ind,l in enumerate(lab):
            if dirList[ind].find(l) != -1:
                Y = ind
                continue
        ## Format Data
        # Max sequence length
        ## Format data into X Y and mask
        X = np.zeros(shape=(1, MAX_LENGTH, N_FEATURE_DIM))
        mask = np.zeros((1, MAX_LENGTH))
        Y = np.zeros((1,))
        X[0,0:len(data),:] = data
        mask[0] = [1]*len(data)+[0]*(MAX_LENGTH-len(data))
        Y = Y.astype('int32')
        ## Save Data
        f = h5py.File(os.path.join(saveDir,os.path.splitext(dirList[i])[0]+'.h5'),'w')
        f.create_dataset('data',data=X)
        f.create_dataset('label',data=Y)
        f.create_dataset('mask',data=mask)
        f.close()
    #    print('Formating data DONE')


if __name__ == '__main__':
    print read_htk('/home/james/data/mfcc_all/Alaskan_male_5_stand_JJ_20151021_110127.mfc')
