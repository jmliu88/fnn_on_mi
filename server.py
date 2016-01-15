import socket
import sys
import time
from multiprocessing import Process
import os
import model
import numpy as np

import theano.sandbox.cuda

def tcplink(sock, addr):
    try:
        theano.sandbox.cuda.use('cpu')
        ext = '.txt'
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
            x,m = model.readHTKfeat(data)
            print 'read feature'
            decision = get_output(x,m)
            val_predictions = np.argmax(decision, axis=1)
            final_prediction = lab[val_predictions]
            print val_predictions
            sock.send((' %s' % final_prediction.decode('utf-8')).encode('utf-8'))
    except BaseException,e:
        raise(e)
    finally:
        sock.close()
        print('Connection from %s:%s closed.' % addr)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('',9527))
s.listen(5)
print 'Socket is built.'

nn,get_output = model.build_nn(sys.argv[1])
#get_output = 1
lab = ['FN','BS','XQ','KX','JJ','HP','nobark']
print('Model loaded. Waiting for connection...')

while True:
    sock, addr = s.accept()
    p = Process(target=tcplink, args=(sock, addr))
    p.start()
