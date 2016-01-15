import socket
import sys
import time
from multiprocessing import Process
import os
import model
import numpy as np

import theano.sandbox.cuda
import string
import random
import feature_operation
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def tcplink(sock, addr):
    try:
        theano.sandbox.cuda.use('cpu')
        ext = '.wav'
        print('Accept new connection from %s:%s...' % addr)
        while True:
            data = sock.recv(1024)
            data = data.strip()
            print(data)
            #if not data or os.path.splitext(data)[-1] != ext:
#        if os.path.splitext(data)[-1] != ext:
#            print data
#            print os.path.splitext(data)[-1]
#            sock.send(('File extention should be %s. Connection closed!'%ext))
#            sock.close()
#            return
            tmpmfc = '/tmp/%s.mfc.tmp'%id_generator()
            os.system('HCopy -c %s %s %s'%('wav_confi',data,tmpmfc))
            x,m = feature_operation.read_htk(tmpmfc)
            print 'read feature'
            val_predictions ,final_prediction = mdl.predict(x,m)
            print val_predictions
            sock.send((' %s' % final_prediction.decode('utf-8')).encode('utf-8'))
            os.system('rm %s'%tmpmfc)
    except BaseException,e:
        raise(e)
    finally:
        sock.close()
        print('Connection from %s:%s closed.' % addr)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('',9527))
s.listen(5)
print 'Socket is built.'

#nn,get_output = model.build_nn(sys.argv[1])
mdl = model.Model()
mdl.compile()
mdl.load(sys.argv[1])
#get_output = 1
print('Model loaded. Waiting for connection...')

while True:
    sock, addr = s.accept()
    p = Process(target=tcplink, args=(sock, addr))
    p.start()
