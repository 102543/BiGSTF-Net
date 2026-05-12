import torch
import numpy as np
from load import Dataset, Load_Dataset_fnirs, Load_Dataset_EEG, EnhanceDataset, Load_Dataset_WG
# from model import RegularizedDualModalNet
from model import FNIRS_EEG_T, EEG_BRANCH, FNIRS_BRANCH
from split import Split_Dataset_sub, Split_train_validate
import os
import torch.nn as nn
import torch.nn.functional as F
from config import model_config, loss_config, data_aug_config
import time

class SphereFace(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m  # 角度间隔参数
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, labels):
        # L2归一化
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        # 计算cosθ
        cos_theta = F.linear(x, w)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        
        # 计算cos(mθ)
        theta = torch.acos(cos_theta)
        cos_m_theta = torch.cos(self.m * theta)
        
        # 构建目标one-hot
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # SphereFace公式
        output = cos_theta - one_hot * (cos_theta - cos_m_theta)
        
        # 计算损失
        log_probs = F.log_softmax(output, dim=1)
        loss = - (log_probs * one_hot).sum(dim=1).mean()
        
        return loss

class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class RandomLabelSmoothing(torch.nn.Module):
    """NLL loss with random label smoothing within a range"""
    def __init__(self, min_smoothing=0.02, max_smoothing=0.12):
        super(RandomLabelSmoothing, self).__init__()
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing
        self.a_softmax = SphereFace(256, 2)
    def forward(self, inputs, target):
        if not loss_config.only_softmax:
            x, feat = inputs
            # 每次前向传播时随机生成平滑系数
            if loss_config.normal_label_smooth:
                smoothing = (self.min_smoothing + self.max_smoothing) / 2
            else:
                smoothing = torch.empty(1).uniform_(self.min_smoothing, self.max_smoothing).item()
            confidence = 1.0 - smoothing
            
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = confidence * nll_loss + smoothing * smooth_loss
            if self.training and loss_config.use_asoftmax:
                loss += (0.05 * self.a_softmax(feat, target))
            return loss.mean()
        else:
            return F.cross_entropy(inputs[0], target)

def train_model(net, epoch, criterion, optimizer, train_loader, device):
    net.train()
    criterion.train()
    train_running_acc = 0
    total = 0
    loss_steps = []
    for data in train_loader:
        eeg_x, fnirs_x, fnirs_y = data
        # torch.Size([64, 2, 36, 30]) torch.Size([64])
        # print("--------------> ", inputs.size(), labels.size())
        eeg_x = eeg_x.to(device)
        fnirs_x = fnirs_x.to(device)
        fnirs_y = fnirs_y.to(device)

        outputs, feat = net(eeg_x, fnirs_x)
        loss = criterion((outputs, feat), fnirs_y.long())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # max_norm可根据需要调整
        optimizer.step()

        loss_steps.append(loss.item())
        total += fnirs_y.shape[0]
        pred = outputs.argmax(dim=1, keepdim=True)
        train_running_acc += pred.eq(fnirs_y.view_as(pred)).sum().item()

    train_running_loss = float(np.mean(loss_steps))
    train_running_acc = 100 * train_running_acc / total
    return train_running_loss, train_running_acc

def test_model(net, epoch, criterion, test_loader, device):
    net.eval()
    criterion.eval()
    test_running_acc = 0
    total = 0
    loss_steps = []
    
    # 真实 label 和 预测 label
    real_label = []
    pred_label = []
    
    with torch.no_grad():
        for data in test_loader:
            eeg_x, fnirs_x, fnirs_y = data
            eeg_x = eeg_x.to(device)
            fnirs_x = fnirs_x.to(device)
            fnirs_y = fnirs_y.to(device)
            # print("---------<<<<<", inputs.size())
            outputs, feat = net(eeg_x, fnirs_x)
            loss = criterion((outputs, feat), fnirs_y.long())
            
            # 真实 label
            real_label.extend(fnirs_y.tolist())
            # 预测 label
            pred_label.extend(outputs.argmax(dim=1).tolist())
            
            loss_steps.append(loss.item())
            total += fnirs_y.shape[0]
            pred = outputs.argmax(dim=1, keepdim=True)
            test_running_acc += pred.eq(fnirs_y.view_as(pred)).sum().item()

        test_running_acc = 100 * test_running_acc / total
        test_running_loss = float(np.mean(loss_steps))
    return test_running_loss, test_running_acc, real_label, pred_label


def train(subject, kfold_idx):
    # 预测label 和真实 label
    pred_label = []
    real_label = []
    
    # 取出当前测试者和当前的k折
    X_eeg_train, y_eeg_train, X_eeg_test, y_eeg_test = \
        sub_eeg_train_x[sub][kfold_idx], sub_train_y[sub][kfold_idx], \
        sub_eeg_test_x[sub][kfold_idx], sub_eeg_test_y[sub][kfold_idx]
    
    X_fnirs_train, y_fnirs_train, X_fnirs_test, y_fnirs_test = \
        sub_fnirs_train_x[sub][kfold_idx], sub_train_y[sub][kfold_idx], \
        sub_fnirs_test_x[sub][kfold_idx], sub_fnirs_test_y[sub][kfold_idx]            
    
    X_eeg_validate, y_eeg_validate, X_fnirs_validate, y_fnirs_validate = \
        sub_eeg_validate_x[sub][kfold_idx], sub_valite_y[sub][kfold_idx], \
        sub_fnirs_valite_x[sub][kfold_idx], sub_valite_y[sub][kfold_idx]
    
    path = save_path + '/' + str(sub) + '/' + str(kfold_idx)
    if not os.path.exists(path):
        os.makedirs(path)

    train_set = EnhanceDataset(X_eeg_train, X_fnirs_train, y_eeg_train, True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    validate_set = EnhanceDataset(X_eeg_validate, X_fnirs_validate, y_eeg_validate, True)
    validate_loader = torch.utils.data.DataLoader(
        validate_set,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    test_set = EnhanceDataset(X_eeg_test, X_fnirs_test, y_eeg_test, False)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    # -------------------------------------------------------------------------------------------------------------------- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_config.use_single_branch:
        if model_config.single_branch_name == "eeg":
            net = EEG_BRANCH(n_class=2, sampling_point_eeg = 360, \
                sampling_point_fnirs = 30, dim=128, depth=6, heads=8, mlp_dim=64).to(device)
        else:
            net = FNIRS_BRANCH(n_class=2, sampling_point_eeg = 360, \
                sampling_point_fnirs = 30, dim=128, depth=6, heads=8, mlp_dim=64).to(device)
    else:
        net = FNIRS_EEG_T(n_class=2, sampling_point_eeg = 360, \
                sampling_point_fnirs = 30, dim=128, depth=6, heads=8, mlp_dim=64).to(device)


    criterion = RandomLabelSmoothing().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), weight_decay=1e-2)
    lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    # -------------------------------------------------------------------------------------------------------------------- #
    validate_max_acc, validate_max_acc_epoch = 0, 0
    test_max_acc = 0
    for epoch in range(EPOCH):
        # 训练模型
        train_running_loss, train_running_acc = train_model(net, epoch, criterion, \
            optimizer, train_loader, device)
        print('训练 [%d, %d, %d] Train loss: %0.4f, Train acc: %0.3f%%' % (sub, kfold_idx, \
            epoch, train_running_loss, train_running_acc))

        # 验证模型
        val_running_loss, val_running_acc, cur_real_label, cur_pred_label = test_model(net, epoch, criterion, \
            validate_loader, device)
        print('验证 [%d, %d, %d] val loss: %0.4f, val acc: %0.3f%%' % (sub, kfold_idx, \
            epoch, val_running_loss, val_running_acc))

        if val_running_acc > validate_max_acc:
            validate_max_acc = val_running_acc
            validate_max_acc_epoch = epoch
            # save model
            torch.save(net.state_dict(), path + '/model_{}_{}.pt'.format(sub, kfold_idx))
            # save results
            test_save = open(path + '/test_acc.txt', "w")
            test_save.write("sub=%d, best_acc= %.3f" % (sub, val_running_acc))
            test_save.close()
            
            # 真实 label
            real_label = cur_real_label
            # 预测 label
            pred_label = cur_pred_label
            

        if validate_max_acc_epoch + max_validate_stop_epoch_num <  epoch:
            print("验证集准确率没有提升，停止训练")
            break
        lrStep.step(epoch)
    
    # 合并训练集和验证集    
    combined_train_set = torch.utils.data.ConcatDataset([train_set, validate_set])
    combined_train_loader = torch.utils.data.DataLoader(
        combined_train_set,
        batch_size=64,
        shuffle = True,
        num_workers=4
    )
    # 加载之前早停止的模型
    net.load_state_dict(torch.load(path + '/model_{}_{}.pt'.format(sub, kfold_idx)))
    print("********* 重新加载模型，开始模型微调 ******************")
    for epoch in range(0, EPOCH - 30):
        # 开始训练
        train_running_loss, train_running_acc = train_model(net, epoch, criterion, \
            optimizer, combined_train_loader, device)
        print('微调 [%d, %d, %d] Train loss: %0.4f, Train acc: %0.3f' % (sub, kfold_idx, \
            epoch, train_running_loss, train_running_acc))
        
        # 开始测试
        test_running_loss, test_running_acc, cur_real_label, cur_pred_label = test_model(net, epoch, criterion, \
            test_loader, device)
        print('微调 [%d, %d, %d] Test loss: %0.4f, Test acc: %0.3f' % (sub, kfold_idx, \
            epoch, test_running_loss, test_running_acc))
        
        if test_running_acc > test_max_acc:
            test_max_acc = test_running_acc
            # 真实 label
            real_label = cur_real_label
            # 预测 label
            pred_label = cur_pred_label
            print("For sub %d, kfold %d, test acc update: %f" %(subject, kfold_idx, test_running_acc))

    return test_max_acc, real_label, pred_label

if __name__ == "__main__":
    # Training epochs
    EPOCH = 150
    
    kfold = 3
    
    # 如果连续30个epoch验证集acc没有提升，就停止训练
    max_validate_stop_epoch_num = 30
    
    # Select dataset by setting dataset_id
    dataset = ['MI', "MA", "WG"]
    dataset_id = 0


    # Select model by setting models_id
    models = ['fNIRS-T', 'fNIRS-PreT', "FNIRS_EEG_T"]
    models_id = 2
    print(models[models_id])


    # Select the specified path
    mi_fnirs_data_path = "/home/ubuntu/datasets/predata_MI/"
    mi_eeg_data_path = "/home/ubuntu/datasets/eeg_results/"

    ma_fnirs_data_path = "/home/ubuntu/datasets/MA_DATASET/predata_MA/"
    ma_eeg_data_path = "/home/ubuntu/datasets/MA_DATASET/eeg_results_MA/"

    wg_data_path = "/home/ubuntu/datasets/WG_dataset/no_ica_epoch/"
    
    with open("config.py", "r") as file:
        cfg = file.read()
        print("**************** Train config:  ***********\n", cfg)
            
    # Save file and avoid training file overwriting.
    save_path = './results_{}_{}/'.format(dataset[dataset_id], str(time.time()))
    print("************ result save dir: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load dataset, set flooding levels and number of Subjects. Different models may have different flooding levels.
    if dataset[dataset_id] == 'MI':
        Subjects = 29
        fnirs_feature, fnirs_label = Load_Dataset_fnirs(mi_fnirs_data_path, 100, 200)
        eeg_feature, eeg_label = Load_Dataset_EEG(mi_eeg_data_path)
    elif dataset[dataset_id] == 'MA':
        Subjects = 29
        fnirs_feature, fnirs_label = Load_Dataset_fnirs(ma_fnirs_data_path, 100, 200)
        eeg_feature, eeg_label = Load_Dataset_EEG(ma_eeg_data_path)
    else:
        Subjects = 26
        eeg_feature, fnirs_feature, fnirs_label, eeg_label = Load_Dataset_WG(wg_data_path, 0, 10)
    
    if dataset[dataset_id] == 'MI' or dataset[dataset_id] == 'MA':
        
        # Split dataset to training set and test set.
        sub_eeg_train_x, sub_eeg_train_y, sub_eeg_test_x, sub_eeg_test_y, \
        sub_fnirs_train_x, sub_fnirs_train_y, sub_fnirs_test_x, sub_fnirs_test_y \
            = Split_Dataset_sub(eeg_feature, fnirs_feature, eeg_label, kfold, dataset[dataset_id])
        print("------------********* 数据的样本个数: ", len(sub_eeg_train_x))
        
        sub_eeg_train_x, sub_fnirs_train_x, sub_train_y, \
            sub_eeg_validate_x, sub_fnirs_valite_x, sub_valite_y = Split_train_validate(sub_eeg_train_x, sub_fnirs_train_x, sub_eeg_train_y, 0.2, 29)
    else:
        # Split dataset to training set and test set.
        sub_eeg_train_x, sub_eeg_train_y, sub_eeg_test_x, sub_eeg_test_y, \
        sub_fnirs_train_x, sub_fnirs_train_y, sub_fnirs_test_x, sub_fnirs_test_y \
            = Split_Dataset_sub(eeg_feature, fnirs_feature, eeg_label, kfold, dataset[dataset_id])
        print("------------********* 数据的样本个数: ", len(sub_eeg_train_x))
        
        sub_eeg_train_x, sub_fnirs_train_x, sub_train_y, \
            sub_eeg_validate_x, sub_fnirs_valite_x, sub_valite_y = Split_train_validate(sub_eeg_train_x, sub_fnirs_train_x, sub_eeg_train_y, 0.2, 26)

    test_acc_max_list = []
    all_test_real_label, all_test_pred_label = [], []
    for sub in range(0, Subjects):
        kfold_acc_list = []
        test_real_label, test_pred_label = [], []
        for kfold_idx in range(0, kfold):
            max_test_acc, cur_real_label, cur_pred_label = train(sub, kfold_idx)
            test_real_label.append(cur_real_label)
            test_pred_label.append(cur_pred_label)

            kfold_acc_list.append(max_test_acc)
            print("************ For sub: ", sub, "kfold: ", kfold_idx, "max_test_acc: ", max_test_acc)
        test_acc_max_list.append(kfold_acc_list)
        print("************ For sub: ", sub,  "acc list: ", kfold_acc_list, ", mean acc: ", np.mean(np.asarray(kfold_acc_list)))
        
        all_test_real_label.append(test_real_label)
        all_test_pred_label.append(test_pred_label)
    
    print("训练完成. 平均测试集准确率: ", np.mean(np.asarray(test_acc_max_list)))
    print("训练完成. 测试集准确率: ", test_acc_max_list)
    print("训练完成，测试集标准差: ", np.std(np.asarray(test_acc_max_list).mean(axis = 1)))

    print("测试集真实 label: ", all_test_real_label)
    print("测试集预测 label: ", all_test_pred_label)
