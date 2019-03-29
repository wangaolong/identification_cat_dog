#使用训练train.py得到的比较高迭代次数的模型进行测试
#读取保存的模型,看看模型效果如何!

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse



image_size = 100
num_channels=3
images = []

path = 'dog.1016.jpg'
image = cv2.imread(path) #读取一张图片
# 图像预处理过程和训练过程必须完全一致
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image) #将预处理之后的一张图片添加到images列表中
images = np.array(images, dtype=np.uint8) #将images转换成array格式
images = images.astype('float32') #数据类型转换成float32类型
images = np.multiply(images, 1.0/255.0)  #数据归一化

#输入到网络中的数据形状 [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels) #这里只有一张图片,所以第一维是1

#将保存的session加载进来
sess = tf.Session()
#第一步: 重新创建网络图像. 在这一步中,只创建了图像
saver = tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-2600.meta')
#第二步: 加载网络的参数,使用比较靠后的训练次数,正确率高
saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-2600')

#将我们刚才加载进来的默认图加载进来
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0") #加载进来预测结果(网络输出op)

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")  #加载进来输入op(placeholder)
y_true = graph.get_tensor_by_name("y_true:0")  #加载进来实际结果op(placeholder)
'''
因为用的是加载进来的参数
而那些参数是早就已经训练好的
所以这里feed进去的标签值可以是任意值,所以这里用zeros方法生成了[0 0]
因为我们这里不是在用预测值和真实值进行正确率的计算,所以我们只要知道最后的结果是猫是狗即可
根本不用知道这个标签是什么
'''
y_test_images = np.zeros((1, 2))  #用来测试的图片标签

lr = graph.get_tensor_by_name("learning_rate:0")
k_p = graph.get_tensor_by_name("keep_prob:0")


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images, lr: 1e-4, k_p: 1.0}
'''
这里run这个y_pred,并不是在进行训练,而是将网络使用现成的参数跑了一遍
所以只要feed进去的x_batch是对的就行,y_test_images和lr与否根本不用管
'''
result=sess.run(y_pred, feed_dict=feed_dict_testing) #得到网络输出
# result is of this format [probabiliy_of_rose probability_of_sunflower]
# dog [1 0], cat [0 1]
res_label = ['dog','cat']
print(res_label[result.argmax()])
