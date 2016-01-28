# -*- coding:utf-8 -*-
import socket
import os
#audioDir = '/home/james/data/wav'
audioDir = '/home/james/audio'
fs = os.listdir(audioDir)
for i in fs:#range(19,39):
    if os.path.splitext(i)[-1] != '.wav':
        continue
    print '%s'%i
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 建立连接:
    s.connect(('localhost', 9527))
# 接收欢迎消息:
    #print(s.recv(1024).decode('utf-8'))
    #for data in [b'Michael', b'Tracy', b'Sarah']:
        # 发送数据:

    #s.send(b'%d.wav'%i)
    s.send(i.strip())
    print(s.recv(1024).decode('utf-8'))
    s.send(b'exit')
    s.close()
