# This is another classification algorithm
# Say we have three different attributes
# [A, B, C]
# Given [ax, by], answer about c?
# Different from kNN, here we want to classify [ax, by] into
# the existing class CL(x). Since cz is already found in CL(x),
# we solve the problem.

# ? Storage
from math import log


def majorityCnt(classList):
    classCount = {}
    for e in classList:
        if e not in classCount.keys():
            classCount[e] = 0
        classCount[e] += 1
    
    sortedClassList = classList.sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassList[0][0]

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    # H(X) = Sigma(i:1->n)(-Pr(i)log(2,Pr(i)))
    # When Pr(i) = 0 | 1, H(X) gets the minimum,
    # When Pr(x) = 1/N for every element, H(X) gets the maximum
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for d in dataSet:
        if d[axis] == value:
            reducedDataSet = d[:axis]
            reducedDataSet.extend(d[axis+1:])
            retDataSet.append(reducedDataSet)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataset) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def buildDecisionTree(dataSet):
    class_list = [e[-1] for e in dataSet]
    # All elements are in the same class, return the class value 
    if len(dataSet) == class_list.count(class_list[0]):
        return class_list[0]
    # Run out of any attribute
    # Just vote for the majority
    if len(dataSet[0]) == 1:
        return majorityCnt(class_list)

    # More than one feature, choose the best feature to split
    bestFeature = chooseBestFeatureToSplit(dataSet)
    myTree = {bestFeature: {}}
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeature][value] = buildDecisionTree(splitDataSet(dataSet, bestFeature, value))
    
    return myTree

        
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels



