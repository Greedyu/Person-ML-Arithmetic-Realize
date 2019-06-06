import os
import sys

iplist = list()
ip1 = '10.10.22.1'
# ip = '172.24.186.191'
ip = '10.10.22.15'
backinfo =  os.system('ping -c 3 %s'%ip1) # 实现pingIP地址的功能，-c1指发送报文一次，-w1指等待1秒
backinfo1 =  os.system('ping -c 3  %s'%ip) # 实现pingIP地址的功能，-c1指发送报文一次，-w1指等待1秒
print(backinfo)
print(backinfo1)