#在断点上右键可以下条件断点
#按住ctrl,点击函数调用处,可以看到调用的函数定义
#按住ctrl,点击函数定义,可以看到是谁调用的此函数

import sys
import dataset
import tensorflow as tf

import numpy as np
# conda install --channel https://conda.anaconda.org/menpo opencv3 #下载opencv包

from numpy.random import seed #导入seed方法,设定随机种子,以后的随机都是一样的
seed(10)
from tensorflow import set_random_seed
set_random_seed(20)

batch_size = 32 # 总数据量并不大,这里指定一批次32张图片

classes = ['dogs','cats'] # 标签类别
num_classes = len(classes) # 二分类问题

validation_size = 0.2 #百分之20的验证集,百分之80的训练集,确认没有过拟合
img_size = 100 #图像尽量是方形的,而且一定要一样大
num_channels = 3 #彩色图片的频道数是3
train_path='training_data' #训练集路径

# 我们应该使用openCV加载训练集和验证集,并在训练的时候使用
# 上面import dataset,就是dataset.py中我们定义的一些用于辅助的函数
# 这里使用上面定义的参数来读取train_sets文件夹中图片,并以2:8来划分验证集和训练集
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("输入数据读取完成. 打印一数据长度")
print("训练集数据量:\t{}".format(len(data.train.labels)))
print("验证集数据量:\t{}".format(len(data.valid.labels)))


#会话
session = tf.Session()
#数据
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
#标签
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true') #网络理论结果(one_hot格式)
y_true_cls = tf.argmax(y_true, dimension=1) #shape:(?, ) #预测理论结果(二分类问题,只能是0或1)
#学习率
lr = tf.placeholder(tf.float32, name='learning_rate')
#dropout参数
k_p = tf.placeholder(tf.float32, name='keep_prob')



#网络模型相关参数
filter_size_conv1 = 3 #卷积核size
num_filters_conv1 = 32 #卷积核数量

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

'''
512, 1024, 2048都是很常见的个数
这里取2048
'''
fc_layer_size = 1024 #全连接层神经元个数(准备映射成1024维特征)

def create_weights(shape): #构造权重参数
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size): #构造偏置参数
    return tf.Variable(tf.constant(0.05, shape=[size]))


'''
input: 图像数据x (None, 64, 64, 3)
num_input_channels: 颜色频道 3
conv_filter_size: 卷积核大小 3
num_filters: 卷积核数量 32或64
'''
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    #使用上面定义好的创建权重参数的函数进行权值初始化 (卷积核大小3, 卷积核大小3, 颜色频道3, 卷积核数量32)
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    #使用上面定义好的创建偏置参数的函数进行偏置初始化 (卷积核数量)
    biases = create_biases(num_filters)

    #创建卷积层
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases
    
    layer = tf.nn.relu(layer) #卷积层输出(None, 64, 64, 32)
    
    #卷积之后池化
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    #layer = tf.nn.relu(layer)

    return layer #卷积层输出

    
'''
将多维数据压平成一维的,用于从卷积层到全连接层的过渡
'''
def create_flatten_layer(layer): #(None, 8, 8, 64)
    #得到卷积层输出的shape: (batch_size img_size img_size num_channels)
    layer_shape = layer.get_shape()

    '''
    layer_shape是(None, 8, 8, 64)
    layer_shape[1:4]是(8, 8, 64)
    num_elements()函数是得到元素总个数8*8*64
    相当于将之前单独每张图片的特征点拉直成一层扁平的神经元
    '''
    num_features = layer_shape[1:4].num_elements()

    #将layer的形状重塑成(None, 8*8*64)
    layer = tf.reshape(layer, [-1, num_features])

    return layer #将传进来的卷积层拉直成扁平的神经元 (None, 4096)


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases

    '''
    先dropout再经过激活函数
    '''
    layer=tf.nn.dropout(layer,keep_prob=k_p) #dropout参数
    
    if use_relu:
        layer = tf.nn.relu(layer)
        

    return layer

#第一卷积池化层
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
#第一卷积池化层
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)
#第一卷积池化层
layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

#卷积层和全连接层之间的过渡,将卷积层神经元压扁成全连接层神经元
layer_flat = create_flatten_layer(layer_conv3)
#第一全连接层(将8*8*64映射成1024维特征)
layer_fc1 = create_fc_layer(input=layer_flat, #(None, 4096)
                     # (None, 4096)索引[1:4]的结果是(4096, ),所以这里的num_inputs就是4096
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size, #上面定义的layer_size,1024
                     use_relu=True) #激活函数使用relu

#第二全连接层(由于是输出结果了,所以就不使用relu函数,直接将1024个特征映射成2个特征进行分类)
#网络实际输出(one_hot格式)
layer_fc2 = create_fc_layer(input=layer_fc1, #(None, 1024)
                     num_inputs=fc_layer_size, #1024
                     num_outputs=num_classes, #2
                     use_relu=False)

#实际结果
y_pred = tf.nn.softmax(layer_fc2,name='y_pred') #使用softmax分配概率,网络实际结果(one_hot格式)
y_pred_cls = tf.argmax(y_pred, dimension=1) #预测实际结果(二分类问题,只能是0或1)

session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, #网络实际输出(one_hot格式)
                                                    labels=y_true) #实际结果(one_hot格式)
cost = tf.reduce_mean(cross_entropy) #代价函数
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost) #Adam优化器
correct_prediction = tf.equal(y_pred_cls, y_true_cls) #使用shape为(None, )来进行正确率预测
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 

'''
训练时输出一下过程中的相关信息
'''
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,i):
    acc, learning_rate = session.run(fetches=[accuracy, lr], feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--iterations:{1}--Training Accuracy:{2:>6.1%}, Validation Accuracy:{3:>6.1%},  Validation Loss:{4:.3f}, Learning Rate:{5:.11f}"
    print(msg.format(epoch + 1,i, acc, val_acc, val_loss, learning_rate))
    return acc, val_acc #这两个参数返回

total_iterations = 0 #总迭代批次

'''
saver模块可以保存模型,也可以读取模型
.meta文件存储的是网络结构图
.data文件存储的是网络参数(比较大的文件)
.index文件存储的是索引
'''
saver = tf.train.Saver()

def train(num_iteration):
    global total_iterations #取得全局变量:总迭代批次

    #从总迭代批次,到总迭代批次+迭代批次
    for i in range(total_iterations,
                   total_iterations + num_iteration):
        '''将训练集和验证集的一批次信息取出来'''
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        '''要feed到run中的训练集和验证集数据'''
        feed_dict_tr_with_keep_prob = {x: x_batch, #带dropout,用于训练模型
                        y_true: y_true_batch,
                        lr: 1e-4 * (0.99999**(i-total_iterations)),
                        k_p: 0.2}
        feed_dict_tr = {x: x_batch, #不带dropout,用于看准确率
                           y_true: y_true_batch,
                           lr: 1e-4 * (0.99999 ** (i - total_iterations)),
                           k_p: 1.0}
        feed_dict_val = {x: x_valid_batch, #不带dropout,用于看准确率
                         y_true: y_valid_batch,
                         lr: 1e-4 * (0.99999**(i-total_iterations)),
                         k_p: 1.0}
        '''将训练集的数据和标签放到优化器中训练一次'''
        session.run(optimizer, feed_dict=feed_dict_tr_with_keep_prob)

        if i % int(data.train.num_examples/batch_size) == 0: #当前迭代批次是整数epoch
            val_loss = session.run(cost, feed_dict=feed_dict_val) #使用验证集数据跑cost得到验证误差
            epoch = int(i / int(data.train.num_examples/batch_size))  #整个训练集被训练了多少遍

            acc, val_acc = show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss,i) #展示相关参数
            '''用刚才创建的saver,保存这个session
            文件将会保存在此文件夹中
            文件的后面-数字,就是global_step,也就是训练的批数'''
            saver.save(session, './dogs-cats-model/dog-cat.ckpt',global_step=i) #i是多少批
            if acc > 0.9 and val_acc > 0.9: #模型效果在接收范围内,直接退出
                print('Finish')
                sys.exit(0)


    total_iterations += num_iteration #将这次训练的次数加到总批次上

train(num_iteration=8000) #调用train函数,训练8000批数据
