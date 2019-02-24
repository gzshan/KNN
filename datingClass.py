import numpy as np
import matplotlib.pyplot as plt
from kNN import classify0

""" 以下几个函数：使用KNN算法改进约会网站的配对效果 """

"""将文本文件转换为numpy，准备数据"""
def file2matrix(fileName):
    fr = open(fileName) # 打开文件
    arrayOfLines = fr.readlines() # 按行读取整个文件，得到行的列表
    numberOfLines = len(arrayOfLines) # 得到文件的行数

    returnMat = np.zeros((numberOfLines,3)) # 要返回的数据矩阵
    classLaberVector = [] # 返回数据对应的标签列表

    index = 0
    """解析数据文件到矩阵和列表"""
    for line in arrayOfLines: #依次读取每行
        line =line.strip() # 去掉两端的空白符
        listFromLine = line.split('\t') # 按制表符分割
        returnMat[index,:] = listFromLine[0:3] # 保存数据
        classLaberVector.append(int(listFromLine[-1])) # 保存对应的标签
        index += 1

    return returnMat,classLaberVector

"""数据图形化展示，使用Matplotlib创建散点图"""
def dataShow(datingDataMat,datingLabels):
    fig = plt.figure() # 创建画布
    ax = fig.add_subplot(111) #在画布上画子图，将画布分为1*1，在第一子图上画图
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
    plt.show()

"""数据的归一化，将数字转化到0和1之间"""
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 取每列的最小值
    maxVals = dataSet.max(0) # 取每列的最大值
    ranges = maxVals - minVals

    normDataSet = np.zeros(np.shape(dataSet)) # 归一化后的数据矩阵
    m = dataSet.shape[0] # 数据个数
    normDataSet = dataSet - np.tile(minVals,(m,1)) # 在行方向上重复m次
    normDataSet = normDataSet / np.tile(ranges,(m,1)) # 归一化
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') # 数据加载
    normDataSet, ranges, minVals = autoNorm(datingDataMat) # 归一化
    m = normDataSet.shape[0] # 数据个数
    numTestVecs = int(m * hoRatio) # 10%的测试数据
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3) # 使用KNN分类
        print("the classifier came back with: %d ,the real answer is: %d " % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

#datingClassTest()

"""约会网站预测函数，构建完整系统"""
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percent of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice Cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 数据加载
    normDataSet, ranges, minVals = autoNorm(datingDataMat)  # 归一化
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normDataSet,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult-1])

if __name__ == '__main__':
    classifyPerson()

"""
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#print(datingDataMat,datingLabels[0:20])
#dataShow(datingDataMat,datingLabels)
normDataSet,ranges,minVals = autoNorm(datingDataMat)
print(normDataSet)
"""