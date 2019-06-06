# -*- coding: UTF-8 -*-
import operator
import random

import jieba
import os


def TextProcessing(folder_path,test_size = 0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        for file in files:
            with open(os.path.join(new_folder_path,file) ,'r',encoding='utf-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw , cut_all= False)
            word_list = list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)

    data_class_list = list(zip(data_list,class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index :]
    test_list = data_class_list[:index]
    train_data_list , train_class_list = zip(*train_list)
    test_data_list , test_class_list = zip(*test_list)

    all_words_dict ={}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    all_words_tuple_list = sorted(all_words_dict.items(),key=operator.itemgetter(1),reverse= True)
    # all_words_tuple_list = sorted(all_words_dict,lambda f:f[1],reverse= True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    print()

if __name__ == '__main__':
    folder_path = './SogouC/Sample'  # 训练集存放地址
    TextProcessing(folder_path)