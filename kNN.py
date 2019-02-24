import numpy as np
import operator

"""模拟创建数据集和对应的标签"""
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

"""KNN算法具体实现"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # 数据个数
    diffMat =np.tile(inX,(dataSetSize,1)) - dataSet # tile对inX在行方向重复dataSetSize次
    sqDiffMat = diffMat ** 2 # 求差的平方
    sqDistances = sqDiffMat.sum(axis=1)  # 平方之和
    distances = sqDistances ** 0.5 # 平方和开根号即为欧式距离
    sortedDistIndicies = distances.argsort() # argsort函数返回的是数组值从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # 统计前k个中标签的数目

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group,labels=createDataSet()
    print(group,labels)
    print(classify0([0,0],group,labels,3))



