import torch
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import *

import argparse
import warnings
import copy

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='nsl')
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--epoch_1", type=int, default=1)
parser.add_argument("--percent", type=float, default=0.8)
parser.add_argument("--conf_percentile", type=float, default=0.8)
parser.add_argument("--min_keep", type=int, default=64)
parser.add_argument("--sample_interval", type=int, default=2000)
parser.add_argument("--cuda", type=str, default="0")

args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
conf_percentile = args.conf_percentile
min_keep = args.min_keep
sample_interval = args.sample_interval
cuda_num = args.cuda

tem = 0.02
bs = 128
seed = 5009
seed_round = 5

if dataset == 'nsl':
    input_dim = 121
else:
    input_dim = 196

if dataset == 'nsl':
    KDDTrain_dataset_path   = "NSL_pre_data/PKDDTrain+.csv"
    KDDTest_dataset_path    = "NSL_pre_data/PKDDTest+.csv"

    KDDTrain   =  load_data(KDDTrain_dataset_path)
    KDDTest    =  load_data(KDDTest_dataset_path)

    # 'labels2' means normal and abnormal, 'labels9' means 'attack_seen', 'attack_unseen', and normal
    # Create an instance of SplitData for 'nsl'
    splitter_nsl = SplitData(dataset='nsl')
    # Transform the data
    x_train, y_train = splitter_nsl.transform(KDDTrain, labels='labels2')
    x_test, y_test = splitter_nsl.transform(KDDTest, labels='labels2')
else:
    UNSWTrain_dataset_path   = "UNSW_pre_data/UNSWTrain.csv"
    UNSWTest_dataset_path    = "UNSW_pre_data/UNSWTest.csv"

    UNSWTrain   =  load_data(UNSWTrain_dataset_path)
    UNSWTest    =  load_data(UNSWTest_dataset_path)

    # Create an instance of SplitData for 'unsw'
    splitter_unsw = SplitData(dataset='unsw')

    # Transform the data
    x_train, y_train = splitter_unsw.transform(UNSWTrain, labels='label')
    x_test, y_test = splitter_unsw.transform(UNSWTest, labels='label')

# Convert to torch tensors
x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)

device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")

criterion = CRCLoss(device, tem)

for i in range(seed_round):
    # Set the seed for the random number generator for this iteration
    setup_seed(seed+i)

    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(x_train, y_train, test_size=percent, random_state=seed+i)
    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=bs, shuffle=True)
    
    num_of_first_train = online_x_train.shape[0]

    model = AE(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

    model.train()
    for epoch in range(epochs):
        print('seed = ', (seed+i), ', first round: epoch = ', epoch)
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)

            labels = labels.to(device)
            optimizer.zero_grad()

            features, recon_vec = model(inputs)
            loss = criterion(features,labels) + criterion(recon_vec,labels)

            loss.backward()
            optimizer.step()
            
    teacher_model = copy.deepcopy(model).to(device)
    teacher_model.eval()
    ema_decay = 0.999

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    online_x_train, online_y_train  = online_x_train.to(device), online_y_train.to(device)

    x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = online_x_train.clone(), online_x_test.clone().to(device), online_y_train.clone(), online_y_test.clone()

####################### start online training #######################
    count = 0
    y_train_detection = y_train_this_epoch
    while len(x_test_left_epoch) > 0:
        print('seed = ', (seed+i), ', i = ', count)
        count += 1
        
        if len(x_test_left_epoch) < sample_interval:
            x_test_this_epoch = x_test_left_epoch.clone()
            x_test_left_epoch.resize_(0)
        else:
            x_test_this_epoch = x_test_left_epoch[:sample_interval].clone()
            x_test_left_epoch = x_test_left_epoch[sample_interval:]

        test_features = F.normalize(model(x_test_this_epoch)[0], p=2, dim=1)

        # must compute the normal_temp and normal_recon_temp again, because the model has been updated
        normal_temp = torch.mean(
            F.normalize(model(online_x_train[(online_y_train == 0).squeeze()])[0], p=2, dim=1),
            dim=0
        )
        normal_recon_temp = torch.mean(
            F.normalize(model(online_x_train[(online_y_train == 0).squeeze()])[1], p=2, dim=1),
            dim=0
        )

        predict_label, predict_conf, predict_src = evaluate(
            normal_temp,
            normal_recon_temp,
            x_train_this_epoch,
            y_train_detection,
            x_test_this_epoch,
            0,
            teacher_model,
            get_confidence=True
        )
        
        predict_label = predict_label.to(device)
        predict_conf = predict_conf.to(device)

        # ===== 只保留高置信伪标签 =====
        conf_threshold = torch.quantile(predict_conf, conf_percentile)
        keep_mask = predict_conf >= conf_threshold

        # 防止筛完一个都不剩
        if keep_mask.sum().item() < min_keep:
            topk = min(min_keep, predict_conf.shape[0])
            _, topk_idx = torch.topk(predict_conf, k=topk, largest=True)
            keep_mask = torch.zeros_like(predict_conf, dtype=torch.bool)
            keep_mask[topk_idx] = True

        x_keep = x_test_this_epoch[keep_mask]
        y_keep = predict_label[keep_mask]

        # detection 分支只追加高置信标签
        y_train_detection = torch.cat((y_train_detection.to(device), y_keep))

        # 训练集也只追加高置信样本
        x_train_this_epoch = torch.cat((x_train_this_epoch.to(device), x_keep.to(device)))
        y_train_this_epoch = torch.cat((y_train_this_epoch.to(device), y_keep))

        train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds, batch_size=bs, shuffle=True)
        model.train()
        for epoch in range(epoch_1):
            print('epoch = ', epoch)
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)

                labels = labels.to(device)
                optimizer.zero_grad()

                features, recon_vec = model(inputs)

                loss = criterion(features,labels) + criterion(recon_vec,labels)

                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            for t_param, s_param in zip(teacher_model.parameters(), model.parameters()):
                t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)
        teacher_model.eval()

################### test the performance after online training ###################
    normal_temp = torch.mean(F.normalize(model(online_x_train[(online_y_train == 0).squeeze()])[0], p=2, dim=1), dim=0)
    normal_recon_temp = torch.mean(F.normalize(model(online_x_train[(online_y_train == 0).squeeze()])[1], p=2, dim=1), dim=0)

    res_en, res_de, res_final = evaluate(normal_temp, normal_recon_temp, x_train_this_epoch, y_train_detection, x_test, y_test, model)
