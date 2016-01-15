# -*- coding:utf-8 -*-
import socket
for i in range(19):
    print 'time:%d'%i
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 建立连接:
    s.connect(('localhost', 9527))
# 接收欢迎消息:
    #print(s.recv(1024).decode('utf-8'))
    #for data in [b'Michael', b'Tracy', b'Sarah']:
        # 发送数据:
    
    s.send(b'%d.wav'%i)
    print(s.recv(1024).decode('utf-8'))
    s.send(b'exit')
    s.close()
