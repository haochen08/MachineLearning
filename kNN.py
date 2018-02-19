from numpy import *
import operator


def createDataSet():
	groups = array([[1.0, 1.1], [1.0, 0.9], [0.1, 0.2], [0.0, 0.1]])
	labels = ['a','a','b','b']
	return groups, labels

def normDataSet(dataSet):
	ranges = dataSet.max(0) - dataSet.min(0)
	dataSet /= ranges
	return dataSet

def realDataSet():
	file = open('datingTestSet.txt', 'r')
	noOfLines = len(file.readlines())

	dataSet = zeros((noOfLines, 3))
	labels = []
	file = open('datingTestSet.txt', 'r')
	for idx, line in enumerate(file.readlines()):
		data = line.split("\t")
		dataSet[idx] = data[0:-1]
		labels.append(data[-1][0:-1]) # Trim the last \n char
	
	file.close()
	return normDataSet(dataSet), labels

def classify(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	# [inx] => [inx, inx, ..., inx]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	# Element-wise multiplication : [x, y]*[x, y] = x^2 + y^2
	# axis 0: row dir; axis 1: column dir
	# [[x0,y0],[x1, y1]].sum(axis=0) = [(x0+x1), (y0+y1)]
	# [[x0,y0],[x1, y1]].sum(axis=1) = [(x0+y0), (x1+y1)]
	distanceSqr = (diffMat * diffMat).sum(axis=1)
	sortedIndices = distanceSqr.argsort()
	classCount = {}
	for i in range(k):
		votedLabel = labels[sortedIndices[i]]
		classCount[votedLabel] = classCount.get(votedLabel, 0)+1
	# sortedClassCount is like (('a', 4), ('b', 3),...)
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def datingClassTest():
	hoRatio = 0.1
	dataSet, labels = realDataSet()
	m = dataSet.shape[0] # 0 is row
	numTestVecs = int(m * hoRatio)
	errorCnt = 0
	for i in range(numTestVecs):
		res = classify(dataSet[i], dataSet[numTestVecs:m], labels[numTestVecs:m], 3)
		print "the classifier came back with: %s, the real answer is: %s" % (res, labels[i])
		if (res != labels[i]): 
			errorCnt += 1
	print "Error rate is: %f" % (errorCnt/float(m))

if __name__ == "__main__":
	datingClassTest()




