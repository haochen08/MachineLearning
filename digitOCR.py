import os
from numpy import *
from kNN import *

TRAINING_DATA_PATH = "trainingDigits"
TEST_DATA_PATH = "testDigits"

IMG_BIT_SIZE = 1024

def img2vec(file):
    lines = file.readlines()  
    vec = []
    data = ""
    for line in lines:
        data += line.rstrip("\r\n")

    for str in data:
        vec.append(int(str))
    return vec

def loadDataSet():
    file_name_list = os.listdir(TRAINING_DATA_PATH)
    m = len(file_name_list)

    data_set = zeros((m, IMG_BIT_SIZE))
    labels = []
    for index, fname in enumerate(file_name_list):
        # {digit}_{index}.txt
        digit = fname.split("_")[0]
        labels.append(digit)
        path = TRAINING_DATA_PATH + "/" + fname
        file = open(path)
        data_set[index] = img2vec(file)
        file.close()
        # No need to normalize the dataSet 
        # as each dataset is like [0,1,0,1,...]
    
    return data_set, labels

def identifyDigit():
    data_set, labels = loadDataSet()
    test_file_name_list = os.listdir(TEST_DATA_PATH)
    error_cnt = 0
    for fname in test_file_name_list:
        digit = fname.split("_")[0]
        path = TEST_DATA_PATH + "/" + fname
        file = open(path)
        vec = img2vec(file)
        file.close()
        res = classify(vec, data_set, labels, 3)
        #print "Machine think it is %s, actually it is %s" % (res, digit)
        if res != digit:
            error_cnt+=1

    print "Error rate: %.2f%%" % (error_cnt * 1.0 / len(test_file_name_list) * 100)


if __name__ == "__main__":
    identifyDigit()



