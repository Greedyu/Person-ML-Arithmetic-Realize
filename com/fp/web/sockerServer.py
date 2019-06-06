# -*- coding:utf-8 -*-

import socket, select

import time



if __name__ == '__main__':
    #   创建套接字
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #   设置IP和端口
    host = 'localhost'
    port = 9999
    #   bind绑定该端口
    mySocket.bind((host, port))
    #   监听
    mySocket.listen(10)

    while True:
        #   接收客户端连接
        print("等待连接....")
        client, address = mySocket.accept()
        print("新连接")
        print("IP is %s" % address[0])
        print("port is %d\n" % address[1])

        while True:
            #   发送消息
            msg = "----------------------send:"
            client.sendto(msg.encode(),(host,port))
            print("发送完成")
            # 读取消息
            time.sleep(0.5)

