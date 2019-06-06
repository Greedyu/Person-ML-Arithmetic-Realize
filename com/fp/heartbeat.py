#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.header import Header
import traceback
import datetime
import time

import os
from sys import argv

log_dir = '/data/logs/python/heartbeat/'

def sendmail():
    sender = 'gre_edyu@163.com'
    password = 'as4658564864'
    receivers = ['602747844@qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '无法ping接大数据服务器...', 'plain', 'utf-8')
    message['From'] = Header("gre_edyu@163.com")  # 发送者
    message['To'] = Header("602747844@qq.com")  # 接收者

    subject = '服务器无法链接 ！！'
    message['Subject'] = Header(subject, 'utf-8')
    try:
        # 163比较特殊，在正式服务器只能用 SMTP_SSL 不能用SMTP,SMTP端口25
        smtpObj = smtplib.SMTP_SSL('smtp.163.com', 465)
        smtpObj.set_debuglevel(1)
        smtpObj.login(sender, password)
        smtpObj.sendmail(sender, receivers, message.as_string())
        smtpObj.quit()
        print('邮件发送成功')
    except Exception as e:
        traceback.print_exc()
        print("Error: 无法发送邮件")


def connect():
    target_ip = '10.10.22.1'
    this_day = datetime.datetime.now().strftime('%Y-%m-%d')
    os.system('echo ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' >> ' + log_dir + this_day + '.log')
    ping_db = os.system('ping -c 3 -W 10 %s >> %s%s.log' % (target_ip, log_dir, this_day))
    return ping_db

def start():

    print('----开始执行心跳检测服务----')
    while(1):
        count = 0
        for index in range(0, 3):
            ping_db = connect()
            if(ping_db == 0):
                break
            count +=1
            time.sleep(5)
        print(count)
        if(count < 3):
            print('----链接成功----')
            time.sleep(10)
        else:
            sendmail()
            print('----链接失败 开始睡眠----')
            time.sleep(60 * 60 * 12)

def stop():
    print('开始停止')
    pid = """` ps -ef |grep heartbeat|grep -w start|grep -v grep|awk '{print $2}' `"""
    os.system('kill -9 '+ pid)



if __name__ == '__main__':
    print(argv[0])
    if((len(argv)) < 2):
        print('请输入有效参数')
    else:
        operate = argv[1]
        if(operate == 'stop'):
            stop()
        elif(operate == 'start'):
            start()
        else:
            print('请输入正确参数')

