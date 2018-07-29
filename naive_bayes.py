import re

'''
 In training set, we know the result for P(W|C(i)) given W by
 Pai(P(W(i)) if for any i and j, W(i) is statistically independent with W(j)

 For a new W, we can answer the question what C(i?) it belongs to.
 Since we can choose the maximum P(C(i)|W) among i.
 P(C(i)|w) = P(W|C(i))P(C(i))/P(W)

 In practice, we compute log(P(W|C(i))) + log(P(C(i))) to avoid underflow to 0 by multiplying two
 very small value (from P(W|C))
'''

def loadDataSet():
    data_set = [ "It is stupid to buy such stuff", "It drives me to mad how can I behave so foolish"]
    class_res = [1,1] # Two classes so far
    return data_set, class_res

def createVocabularyList(dataSet):
    res = set([])
    for e in dataSet:
        words = re.split(" ", e)
        res = res | set(words)
    return list(res)

def convertInputSetToVec(vocalList, inputSet):
    vec = [0]*len(vocalList)
    for e in inputSet:
        words = re.split(" ", e)
        for w in words:
            if w in vocalList:
                vec[vocalList.index(w)] = 1
    return vec
    

def main():
    input_set = ["Doing such stuff is very stupid", "I must be mad to behave like this"]
    data_set, class_res = loadDataSet()
    vocal_list = createVocabularyList(data_set)
    vec = convertInputSetToVec(vocal_list, input_set)

    print vec

if __name__ == '__main__':
    main()