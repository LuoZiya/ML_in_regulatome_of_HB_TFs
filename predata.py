import pandas as pd
import numpy as np


def predata(config, logger):
    # 数据载入
    data = pd.read_csv(f"./data/{config.dataset}")
    data_x, data_y = np.array(data)[:, 0], np.array(data)[:, 1]
    # 分别获取标签为0和1的数据索引
    indices_0 = np.where(data_y == 0)[0]
    indices_1 = np.where(data_y == 1)[0]
    np.random.seed(config.seed)
    # 随机排列索引
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)
    # 按照80%训练，10%验证，10%测试划分数据
    s1_0 = int(len(indices_0) * 0.95)
    s1_1 = int(len(indices_1) * 0.95)
    logger.info(
        f'Baseline：{max(len(indices_1) / (len(indices_0) + len(indices_1)), 1 - len(indices_1) / (len(indices_0) + len(indices_1)))}')
    print(indices_0[:s1_0].shape, indices_1[:s1_1].shape)
    # bs = len(indices_0)//len(indices_1)
    # for i in range(bs):
    #     if i == 0:
    #         indices_my = np.concatenate([indices_0[:s1_0], indices_1[:s1_1]])
    #     else:
    #         indices_my = np.concatenate([indices_my, indices_1[:s1_1]])
    # train_indices = indices_my
    # print(indices_0.shape, indices_1.shape, train_indices.shape)

    train_indices = np.concatenate([indices_0[:int(s1_0 * 0.7)], indices_1[:s1_1]])
    # train_indices = np.concatenate([train_indices, indices_1[:s1_1]])
    # train_indices = np.concatenate([train_indices, indices_1[:s1_1]])
    # train_indices = np.concatenate([train_indices, indices_1[:s1_1]])
    # train_indices = np.concatenate([train_indices, indices_1[:s1_1]])
    # train_indices = np.concatenate([train_indices, indices_1[:s1_1]])
    # train_indices = np.concatenate([train_indices, indices_1[:s1_1]])

    test_indices = np.concatenate([indices_0[s1_0:], indices_1[s1_1:]])
    # 随机打乱索引
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    # 根据索引划分数据集
    train_x, test_x = data_x[train_indices], data_x[test_indices]
    train_y, test_y = data_y[train_indices], data_y[test_indices]
    logger.info(f'总数据量：{len(train_y) + len(test_y)}  train_len: {len(train_y)} test_len: {len(test_y)}')

    train_y = np.array(train_y).astype(int)

    test_y = np.array(test_y).astype(int)

    del data_x, data_y
    logger.info(f'总数据量：{len(train_y)  + len(test_y)} '
                f'train_len: {len(train_y)} test_len: {len(test_y)}')
    return train_x, train_y, test_x, test_y

