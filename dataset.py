#图片文件可以是像素点文件
#我们这里使用了最简单的一种形式,就是.jpg文件的形式
#读取图片的时候,使用openCV进行读取

import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

#alt+左键 回溯到调用此函数处
#alt+右键 从调用此函数处回到此函数处
#下面console右边的三道横杠是跳转到当前debug执行到的语句处
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = [] #类别

    #print的东西可以在底下console看到
    print('Going to read training images')
    for fields in classes:   #有几个分类就循环几次
        index = classes.index(fields) #取出狗和猫对应的索引0和1
        print('Now going to read {} files (Index: {})'.format(fields, index))
        '''
        os.path.join() : 路径拼接函数
        '''
        path = os.path.join(train_path, fields, '*g') #取出training_data中dogs文件夹的所有文件
        '''
        import glob
        
        #获取指定目录下的所有图片
        print (glob.glob(r"/home/qiaoyunhao/*/*.png"),"\n")#加上r让字符串不转义
        
        #获取上级目录的所有.py文件
        print (glob.glob(r'../*.py')) #相对路径
        '''
        files = glob.glob(path) #取得training_data中dogs文件夹中所有文件路径
        '''
        通过glob取得的files并不是从0到500,而是按照字母数字的顺序排列的
        也就是说这里的fl并不是从第1张图片顺序取到第500张的,而是0,1,10,100,101,102...的顺序来取得的
        '''
        for fl in files: #遍历dogs或cats文件夹中所有图片路径
            '''
            这行语句image是三维数组,是某一张图片的(图像宽,图像长,color_channel)
            比如第一张图片是374*499*3
            '''
            image = cv2.imread(fl) #通过路径把图读进来
            '''
            imread进来的图像是三维数组,resize成64*64*3的图像
            之所以resize成64的是为了训练快速
            '''
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32) #将图片像素点数据转化成float32类型
            image = np.multiply(image, 1.0 / 255.0) #每个像素点值都在0到255之间,进行归一化
            images.append(image) #将经过处理的一张图片添加到列表中
            label = np.zeros(len(classes))
            label[index] = 1.0 #为这张图片生成一个label,狗是[1,0],猫是[0,1]
            labels.append(label) #将图片标签添加到labels中(one_hot格式的标签)
            '''
            按照路径得到文件名'dog.0.jpg'等
            '''
            flbase = os.path.basename(fl)
            img_names.append(flbase) #将图片文件名添加到img_names列表中
            cls.append(fields) #将图片类别添加到cls中
    images = np.array(images) #将1000张training_data中猫狗的图片从list转换成ndarray
    labels = np.array(labels) #将1000张training_data中猫狗的标签从list转换成ndarray
    img_names = np.array(img_names) #将1000张training_data中猫狗的文件名从list转换成ndarray
    cls = np.array(cls) #将1000张training_data中猫狗的类别从list转换成ndarray

    return images, labels, img_names, cls


class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0] #传进来的images多大,这里的数据集大小就是多少

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    '''
    #内置的@ property装饰器负责把一个方法变成属性调用的
    #只写getter,不写setter,定义只读属性
    
    小例子:
    >>> s = Student()
    >>> s.score = 60 # OK，实际转化为s.set_score(60)
    >>> s.score # OK，实际转化为s.get_score()
    '''

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """从此数据集中返回下一个batch_size大小的样本"""
        start = self._index_in_epoch #这一批次的起始位置
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples: #如果索引超过数据集总数
            # After each epoch we update this
            self._epochs_done += 1 #记录这个数据集跑了几个epoch
            start = 0 #重置批次起始位置
            self._index_in_epoch = batch_size #重置索引
            assert batch_size <= self._num_examples #断言表达式正确,表达式错误抛出异常
        end = self._index_in_epoch #这一批次的终止位置

        '''将一个批次的相关数据返回'''
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

# 读取train_sets文件夹中的图片,并以2:8来划分验证集和训练集
def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object): #内置的一个类,用来返回分割好的训练集和验证集的相关信息
        pass
    data_sets = DataSets() #内置的对象,data_set.train是训练集,data_set.valid是验证集

    '''
    数据读取
    返回的四个参数分别是: training_data文件夹中1000张猫狗图片的
    图像 images(1000,64,64,3)
    标签 labels(1000,2)
    文件名 img_names(1000,)
    标签名 cls(1000,)
    '''
    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    '''
    这个shuffle函数是从sklearn.utils中import来的,用来打乱array顺序
    函数功能是将几个等长的array同比打乱顺序(相同索引的值打成包,不打乱内部顺序)
    
    从对称性的角度考虑,不能先训练猫再训练狗,所以要打乱
    '''
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    '''
    isinstance是Python中的一个内建函数。是用来判断一个对象的变量类型
    '''
    if isinstance(validation_size, float): #按照0.2来划分验证集
        validation_size = int(validation_size * images.shape[0]) #划分验证集(200个)

    '''
    #从1000张图片中选择索引0到199一共200张作为验证集数据,标签,文件名和类名
    (200,64,64,3) (200,2) (200,) (200,)
    '''
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    '''
    #从1000张图片中选择索引200到999一共800张作为训练集数据,标签,文件名和类名
    (800,64,64,3) (800,2) (800,) (800,)
    '''
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls) #800训练集的相关参数
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls) #200验证集的相关参数

    return data_sets #将分割好的训练集和验证集返回
