
#!/usr/local/bin/python
# coding=utf-8
from flask import Flask,request

app =Flask(__name__)

@app.route('/<test>')
def hello_world(test):
    return test

@app.route('/post',methods=['POST'])
def post_test():
    postValues = request.values
    print(request.values.get('people'))
    print(request.values.get('people').get('name'))
    print(request.values.get('people').get('value'))


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=10000)
