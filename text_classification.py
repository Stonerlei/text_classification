import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
from sklearn.model_selection import KFold


total = 1000
seed = np.zeros([total])
kf = KFold(5, shuffle=False)
train_index = []
test_index = []
for (trn_idx, val_idx) in kf.split(seed):
    train_index.append(trn_idx)
    test_index.append(val_idx)
    

# construct a list
path = "C:\\Users\\Stoner\\Desktop\\大二下课程\\NLP-PPT\\text_classification\\20_newsgroups"
class_names = os.listdir(path)
path = path + "\\"
# print(class_names)
Correct_rate = []
confusion = np.zeros([20,20])
for fold_id in range(5):
    train_set = []
    test_set = []
    for class_name in class_names:
        file_path = path + class_name
        file_names = os.listdir(file_path)
        file_path = file_path + "\\"
        files = []
        for file_name in file_names:
            with open(file_path + file_name, errors="ignore") as file:
                A = file.read()
                files.append(A)
        files = np.asarray(files)
    
        training = files[train_index[fold_id]]
        testing = files[test_index[fold_id]]
        train_set.append(training)
        test_set.append(testing)
    
    # construct the frequency matrix
    # stop words
    stopdir = "C:\\Users\\Stoner\\Desktop\\大二下课程\\NLP-PPT\\text_classification\\english.txt"
    with open(stopdir, errors='ignore') as stopdir:
        stopword = stopdir.read()
    stopword = stopword.splitlines()
    stopword = stopword + ['ain', 'daren', 'hadn', 'mayn', 'mightn', 'mon', 'mustn', 'needn', 'oughtn', 'shan']
    # 将训练集合并成易操作的格式
    seq_train = []
    # print(train_set[0][0])
    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            seq_train.append(train_set[i][j])
    # 构建词袋模型
    cv_train = CountVectorizer(stop_words=stopword, lowercase=True, max_features=20000)
    cv_train_fit = cv_train.fit_transform(seq_train)
    
    freq_matrix_train = cv_train_fit.toarray()
    print(len(freq_matrix_train))
    # print(len(cv_train.get_feature_names()))
    
    seq_test = []
    for i in range(len(test_set)):
        for j in range(len(test_set[i])):
            seq_test.append(test_set[i][j])
    cv_test = CountVectorizer(stop_words=stopword, lowercase=True, vocabulary=cv_train.get_feature_names())
    cv_test_fit = cv_test.fit_transform(seq_test)
    freq_matrix_test = cv_test_fit.toarray()
    
    
    # K-nearest-neighbors
    # 计算k个最近邻
    freq_matrix_train = np.asarray(freq_matrix_train)
    freq_matrix_test = np.asarray(freq_matrix_test)
    k_value = 25
    correct_rate = np.zeros((1, len(test_set)))
    # print(len(freq_matrix_train))
    for i in range(len(freq_matrix_test)):
        k_max = np.zeros((2, k_value))
        for j in range(len(freq_matrix_train)):
            distance = freq_matrix_test[i] * freq_matrix_train[j]
            distance = np.sum(distance)
            # print('distance: ', distance)
            # print(np.argmin(k_max[0]))
            if distance > np.min(k_max[0]):
                k_max[1, np.argmin(k_max[0])] = math.floor(j/total/0.8)  # 与训练集大小有关
                k_max[0, np.argmin(k_max[0])] = distance
        index = np.argsort(-k_max[0])
        temp = np.zeros((1, k_value))
        # print('k_max: ', k_max)
        # print('index: ', index)
        for m in range(k_value):
            # print(k_max)
            temp[0, m] = k_max[1, index[m]]
        # print(temp)
        k_max[0] = -k_max[0]
        k_max[0] = np.sort(k_max[0])
        k_max[0] = -k_max[0]
        k_max[1] = temp
        # print(i, k_max)
        # 加权从最近邻中选出结果
        scores = np.zeros((1, len(test_set)))
        for n in range(k_value):
            # print('k_max: ', k_max[1, n], '取整：', print(int(k_max[1, n])))
            scores[0, int(k_max[1, n])] = scores[0, int(k_max[1, n])] + 1
        confusion[int(i / total/0.2)][np.argmax(scores)] += 0.2
        if np.argmax(scores) == int(i / total/0.2):                    #与训练集大小有关
            correct_rate[0, int(i / total/0.2)] = correct_rate[0, int(i / total/0.2)] + 1/total/0.2
        print(i)
    print('Correct Rate: ', correct_rate)
    print('Average Correct Rate: ', correct_rate.mean())
    Correct_rate.append(correct_rate)
global_average = 0
average = []
for i in range(5):
    global_average = global_average + Correct_rate[i].mean()
    average.append(Correct_rate[i].mean())
global_average = global_average/5
aww = 0
for i in range(20):
    aww += confusion[i][i]/400