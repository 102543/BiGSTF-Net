import numpy as np
import random
import torch

def shuffle_and_interleave(original_list, label):
    # 复制原始列表以避免修改原始数据
    shuffled_list = original_list[:]
    random.shuffle(shuffled_list)
    # 创建两个新列表，用于存储奇数和偶数位置的元素
    odd_list = []
    even_list = []
    # print("label: ", label)
    # 遍历打乱后的列表，分配到奇偶列表中
    for i, value in enumerate(shuffled_list):
        if label[value] % 2 == 0:  # 偶数位置（注意Python中索引从0开始，所以偶数实际上是1, 3, 5...）
            even_list.append(value)
        else:  # 奇数位置
            odd_list.append(value)
    print("result: ", even_list, odd_list, "len: ", len(even_list), len(odd_list))
    # 合并奇偶列表，保持原有的随机顺序
    result = []
    for even, odd in zip(even_list, odd_list):
        result.append(even)
        result.append(odd)
    return result

def split_time(feature, labels, split_point_num, split_step):
    # 存储的结果
    split_feature = []
    split_labels = []
    # test_times 是测试的次数
    test_times = feature.shape[0]

    for i in range(test_times):
        idx = 0
        while idx + split_point_num <= feature.shape[1]:
            split_feature.append(feature[i][idx: idx + split_point_num])
            split_labels.append(labels[i])
            idx += split_step
    # print("------------------->", feature.shape, len(split_feature))
    split_feature = np.stack(split_feature, 0)
    split_labels = np.asarray(split_labels)
    # print("--------------->", split_feature.shape, split_labels.shape)
    return split_feature, split_labels

def Split_Dataset_kFold(feature, label, split_point_num, split_step, idxs: list, kfold = 5, num_per_tester = 60, num_tester = 29):
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(0, num_tester):
        sub_feature = feature[i]
        sub_label = label[i]
        sub_feature = np.transpose(sub_feature, [0, 3, 1, 2])
        
        kfold_train_feature, kfold_test_feature, kfold_train_label, kfold_test_label = [], [], [], []
        step = num_per_tester // kfold
        # 每一个人的 idx
        sub_idxs = idxs[i]
        
        for j in range(0, kfold):
            # 切分训练集和测试集
            test_idx = sub_idxs[j * step: (j + 1) * step]
            test_feature = sub_feature[test_idx]
            test_labels = sub_label[test_idx]
            
            train_idx = list(set(sub_idxs) - set(test_idx))
            train_feature = sub_feature[train_idx]
            train_labels = sub_label[train_idx]
            # print("---------->", train_feature.shape, test_feature.shape)
            # 切分时间片
            train_feature, train_labels = split_time(train_feature, train_labels, split_point_num, split_step)
            test_feature, test_labels = split_time(test_feature, test_labels, split_point_num, split_step)
            # print(train_feature.shape)
            train_feature = train_feature.transpose(0, 2, 3, 1)
            test_feature = test_feature.transpose(0, 2, 3, 1)
            # 插入到列表中
            kfold_train_feature.append(train_feature)
            kfold_test_feature.append(test_feature)
            kfold_train_label.append(train_labels)
            kfold_test_label.append(test_labels)
            
        X_train.append(kfold_train_feature), y_train.append(kfold_train_label)
        X_test.append(kfold_test_feature), y_test.append(kfold_test_label)
    return X_train, y_train, X_test, y_test

def Split_train_validate(eeg_feature, fnirs_feature, label, ratio = 0.2, num_tester = 29):
    eeg_train_x, fnirs_train_x, train_y = [], [], []
    eeg_validate_x, fnirs_valite_x, valite_y = [], [], []
    
    sub_num = len(eeg_feature)
    kfold_num = len(eeg_feature[0])
    
    for i in range(0, num_tester):
        eeg_train_sub_x, fnirs_train_sub_x, train_sub_y = [], [], []
        eeg_validate_sub_x, fnirs_valite_sub_x, valite_sub_y = [], [], []

        for j in range(0, kfold_num):
            sub_eeg_feature = eeg_feature[i][j]
            sub_fnirs_feature = fnirs_feature[i][j]
            sub_label = label[i][j]
            # 切分训练集和验证集
            num = sub_eeg_feature.shape[0]
            idx_list = [ i for i in range(0, num)]
            validate_idx = random.sample(idx_list, k = int(ratio * num))
            train_idx = list(set(idx_list) - set(validate_idx))
            # 切分训练集和验证集
            train_eeg_feature = sub_eeg_feature[train_idx]
            train_fnirs_feature = sub_fnirs_feature[train_idx]
            train_label = sub_label[train_idx]
            
            validate_eeg_feature = sub_eeg_feature[validate_idx]
            validate_fnirs_feature = sub_fnirs_feature[validate_idx]
            validate_label = sub_label[validate_idx]
            # 插入到列表中
            eeg_train_sub_x.append(train_eeg_feature)
            fnirs_train_sub_x.append(train_fnirs_feature)
            train_sub_y.append(train_label)
            
            eeg_validate_sub_x.append(validate_eeg_feature)
            fnirs_valite_sub_x.append(validate_fnirs_feature)
            valite_sub_y.append(validate_label)

        eeg_train_x.append(eeg_train_sub_x)
        fnirs_train_x.append(fnirs_train_sub_x)
        train_y.append(train_sub_y)
        
        eeg_validate_x.append(eeg_validate_sub_x)
        fnirs_valite_x.append(fnirs_valite_sub_x)
        valite_y.append(valite_sub_y)

    return eeg_train_x, fnirs_train_x, train_y, eeg_validate_x, fnirs_valite_x, valite_y

def Split_Dataset_sub(eeg_feature, fnirs_feature, label, kfold = 5, dataset = "MI"):
    """
    切分BCI 中的 MA 和 MI
    """
    # 数据片段时间长度 3s
    data_time = 3
    if dataset == "MA" or dataset == "MI":
        # 采样点数
        eeg_sampling_freq = 120
        fnirs_sampling_freq = 10
        num_tester = 29
    else:
        # 采样点数
        eeg_sampling_freq = 120
        fnirs_sampling_freq = 10
        num_tester = 26
    # 打乱idx
    num_per_tester = 60
    idxs = []
    for i in range(0, num_tester):
        sub_idxs = [i for i in range(0, num_per_tester)]
        # sub_idxs = shuffle_and_interleave(sub_idxs, label[0])
        idxs.append(sub_idxs)
    # print("idxs: ", idxs)
    
    fnirs_train_x, fnirs_train_y, fnirs_test_x, fnirs_test_y = Split_Dataset_kFold(fnirs_feature, label, \
        fnirs_sampling_freq * data_time, fnirs_sampling_freq, idxs, kfold, num_per_tester, num_tester)  
    
    print("fnirs data: -------->", fnirs_train_x[0][0].shape, \
        fnirs_train_y[0][0].shape, fnirs_test_x[0][0].shape, fnirs_test_y[0][0].shape)
    
    eeg_train_x, eeg_train_y, eeg_test_x, eeg_test_y = Split_Dataset_kFold(eeg_feature, label, \
        eeg_sampling_freq * data_time, eeg_sampling_freq, idxs, kfold, num_per_tester, num_tester)
    
    print("eeg data: ===========>", eeg_train_x[0][0].shape, \
        eeg_train_y[0][0].shape, eeg_test_x[0][0].shape, eeg_test_y[0][0].shape)

    return eeg_train_x, eeg_train_y, eeg_test_x, eeg_test_y, \
        fnirs_train_x, fnirs_train_y, fnirs_test_x, fnirs_test_y

if __name__ == "__main__":
    # load fnirs shape: (29, 60, 2, 36, 100) (29, 60)
    # load eeg shape: (29, 60, 1, 30, 1200) (29, 60)
    # eeg_feature =  np.random.random([29, 60, 1, 30, 1200])
    # fnirs_feature = np.random.random([29, 60, 2, 36, 100])
    # labels = np.asarray([i % 2 for i in range(0, 29 * 60)]).reshape([29, 60])
  
    # eeg_train_x, eeg_train_y, eeg_test_x, eeg_test_y, \
    #     fnirs_train_x, fnirs_train_y, fnirs_test_x, fnirs_test_y = Split_Dataset_sub(eeg_feature, \
    #         fnirs_feature, labels, 5)

    # # fnirs data: --------> (384, 2, 36, 30) (384,) (96, 2, 36, 30) (96,)
    # # eeg data: ===========> (384, 1, 30, 360) (384,) (96, 1, 30, 360) (96,)
    # # print(eeg_train_y[0][0]-fnirs_train_y[0][0], eeg_test_y[0][0]-fnirs_test_y[0][0])
    
    # eeg_train_x, fnirs_train_x, train_y, eeg_validate_x, fnirs_valite_x, valite_y = Split_train_validate(eeg_train_x, fnirs_train_x, eeg_train_y, 0.2)
    # print(eeg_train_x[0][0].shape, eeg_train_y[0][0].shape, eeg_validate_x[0][0].shape, valite_y[0][0].shape)
    

    # load fnirs shape: (26, 60, 2, 36, 100) (26, 60)
    # load eeg shape: (26, 60, 1, 28, 1200) (26, 60)
    eeg_feature =  np.random.random([26, 60, 1, 28, 1200])
    fnirs_feature = np.random.random([26, 60, 2, 36, 100])
    labels = np.asarray([i % 2 for i in range(0, 26 * 60)]).reshape([26, 60])
  
    eeg_train_x, eeg_train_y, eeg_test_x, eeg_test_y, \
        fnirs_train_x, fnirs_train_y, fnirs_test_x, fnirs_test_y = Split_Dataset_sub(eeg_feature, \
            fnirs_feature, labels, 5, "WG")

    # fnirs data: --------> (384, 2, 36, 30) (384,) (96, 2, 36, 30) (96,)
    # eeg data: ===========> (384, 1, 28, 360) (384,) (96, 1, 28, 360) (96,)
    # print(eeg_train_y[0][0]-fnirs_train_y[0][0], eeg_test_y[0][0]-fnirs_test_y[0][0])
    
    eeg_train_x, fnirs_train_x, train_y, eeg_validate_x, fnirs_valite_x, valite_y = Split_train_validate(eeg_train_x, fnirs_train_x, eeg_train_y, 0.2, 26)
    print(eeg_train_x[0][0].shape, eeg_train_y[0][0].shape, eeg_validate_x[0][0].shape, valite_y[0][0].shape)