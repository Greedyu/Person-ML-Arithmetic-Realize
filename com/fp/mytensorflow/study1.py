import tensorflow as tf


# matrix1 = tf.constant([[3.,2.]])
# matrix2 = tf.constant([[3.],[2.]])
# sess = tf.Session()
# 使用GPU配置
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#   with tf.device("/GPU:1"):
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.],[2.]])
#     product = tf.matmul(matrix1, matrix2)
#     print(sess.run(product))

# print(sess.run(tf.matmul(matrix1,matrix2)))
sess1 = tf.Session()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x'
# x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
# sub = tf.sub(x, a)
# print(sub.eval())

state = tf.Variable(0,name="counter")
print(state)

line = 'test 32323'
print(line.index('est'))
# print(line.index('esdfs'))
print(line.__contains__('esdfs'))
print(line.__contains__('est'))