import pandas as pd
import numpy as np
# import h5py
#   Sklearn.metrics 机器学习各种评价指标
from sklearn.metrics.pairwise import cosine_similarity as cos
#       pairwise_distances 计算两个矩阵样本之间的距离，即成对距离
from sklearn.metrics import pairwise_distances as pair
#正则化是将样本在向量空间模型上的一个转换，经常被使用在分类与聚类中。
# 函数normalize 提供了一个快速有简单的方式在一个单向量上来实现这正则化的功能。
# 正则化有l1,l2等，这些都可以用上
from sklearn.preprocessing import normalize

# 构建不同的邻居数量，即为K的数量
topk = 10                                #  tiny-imageNet   heat所占内存83gb   83.5gb  cos

# 直接传入每个结点的特征信息和 label标签，方法采用热核方式
def construct_graph(features, label, method='heat'):
    # fname = 'graph/uspsCC5_graph.txt'  # 创建K近邻为1的文件，若有则覆盖惹
    fname = 'graph8/imageNet-dogs10_graph.txt'
    num = len(label) # 获得标签的长度
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos': #多做一次正则化
        #将每一个大于0的特征全部变成1
        features[features > 0] = 1
        # normalize正则化是将样本在向量空间模型上的一个转换，经常被使用在分类与聚类中
        # axis=1表示对每一行去做这个操作，axis=0表示对每一列做相同的这个操作
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []      #是输入0显示多少行，也就是样本个数
    for i in range(dist.shape[0]): #第i行全部元素
        #           dist[i,:] 第一行，全部列，也就是第一行的全部元素
        #     -(topk+1),topk若为1，则此处值为-2，意思是找到数组前第二大的值，因为第一大的是该对象本身和自己的cos值最大，
        #               topk若为5，则此处值为-6，意思是找到数组前5大的值
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind) # 将找到的 样本配对写入数组当中，从0样本开始[3300,0] [3510,1]......

        f = open(fname, 'w')
    counter = 0
    # 函数主要是想实现构造一个矩阵A，其维度与矩阵dist一致，并为其初始化为全0
    A = np.zeros_like(dist)
    # 遍历刚刚得到的样本配对数组inds
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:   # 因为数组里面的排序不同，所以遇到样本序号和数组序号相同的即不用检测即可
                pass
            else:
                if label[vv] != label[i]: #计算类别不一致，也就是错误的配对情况
                    counter += 1    #  与数组中序号不同的样本写入文件
                f.write('{} {}\n'.format(i, vv)) # 写入文件
    f.close()
    print('+: {}'.format(counter / (num * topk))) #打印错误率

'''
f = h5py.File('data/usps.h5', 'r')
train = f.get('train')
test = f.get('test')
X_tr = train.get('data')[:]
y_tr = train.get('target')[:]
X_te = test.get('data')[:]
y_te = test.get('target')[:]
f.close()
usps = np.concatenate((X_tr, X_te)).astype(np.float32)
label = np.concatenate((y_tr, y_te)).astype(np.int32)
'''

'''
hhar = np.loadtxt('data/hhar.txt', dtype=float)
label = np.loadtxt('data/hhar_label.txt', dtype=int)
'''
#
# reut = np.loadtxt('data/reut.txt', dtype=float)
# label = np.loadtxt('data/reut_label.txt', dtype=int)
#
# uspsCC = np.loadtxt('data/usps_cc.txt',dtype=float)
# uspslabel = np.loadtxt('data/usps_label.txt',dtype=int)
#
# flowers17 = np.loadtxt('data/flowers17.txt',dtype=float)
# flowers17label = np.loadtxt('data/flowers17_label.txt',dtype=float)
#
# official_stl10zn = np.loadtxt('data3/official_stl10zn.txt',dtype=float)
# official_stl10zn_label = np.loadtxt('data3/official_stl10zn_label.txt',dtype=float)
#
# official_cifar10 = np.loadtxt('data3/official_cifar10.txt',dtype=float)
# official_cifar10_label = np.loadtxt('data3/official_cifar10_label.txt',dtype=float)
#
# scene15 = np.loadtxt('data2/scene15.txt',dtype=float)
# scene15label = np.loadtxt('data2/scene15_label.txt',dtype=float)
#
# cub200 = np.loadtxt('data2/cub200.txt',dtype=float)
# cub200label = np.loadtxt('data2/cub200_label.txt',dtype=float)
#
# imageNet10 = np.loadtxt('data4/imageNet-10.txt',dtype=float)
# imageNet10label = np.loadtxt('data4/imageNet-10_label.txt',dtype=float)
#
# imageNetdogs = np.loadtxt('data2/imageNet-dogs.txt',dtype=float)
# imageNetdogslabel = np.loadtxt('data2/imageNet-dogs_label.txt',dtype=float)
#
# official_cifar10 = np.loadtxt('data3/official_cifar10.txt',dtype=float)
# official_cifar10_label = np.loadtxt('data3/official_cifar10_label.txt',dtype=float)
#
# official_cifar20 = np.loadtxt('data3/official_cifar20.txt',dtype=float)
# official_cifar20_label = np.loadtxt('data3/official_cifar20_label.txt',dtype=float)
#
# cifar10 = np.loadtxt('data3/cifar10.txt',dtype=float)
# cifar10_label = np.loadtxt('data3/cifar10_label.txt',dtype=float)
#
# cifar20 = np.loadtxt('data2/cifar20.txt',dtype=float)
# cifar20_label = np.loadtxt('data2/cifar20_label.txt',dtype=float)
#
# cifar100 = np.loadtxt('data5/cifar100.txt',dtype=float)
# cifar100_label = np.loadtxt('data5/cifar100_label.txt',dtype=float)
#
#
# tiny_ImageNet = np.loadtxt('data5/tiny_ImageNet.txt',dtype=float)
# tiny_ImageNet_label = np.loadtxt('data5/tiny_ImageNet_label.txt',dtype=float)
#
#

# imageNetdogs7 = np.loadtxt('data7/imageNet-dogs.txt',dtype=float)
# imageNetdogslabel7 = np.loadtxt('data7/imageNet-dogs_label.txt',dtype=float)
#
imageNetdogs8 = np.loadtxt('data8/imageNet-dogs.txt',dtype=float)
imageNetdogslabel8 = np.loadtxt('data8/imageNet-dogs_label.txt',dtype=float)

# imageNetdogs9 = np.loadtxt('data9/imageNet-dogs.txt',dtype=float)
# imageNetdogslabel9 = np.loadtxt('data9/imageNet-dogs_label.txt',dtype=float)



if __name__ == "__main__":
    print('fsfdsfds')
    print()
    # construct_graph(reut, label, 'ncos')
    construct_graph(imageNetdogs8, imageNetdogslabel8, 'heat')
