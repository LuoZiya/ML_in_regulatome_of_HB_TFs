import logging
import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def bulit_logger(file_path):
    """制作一个Python日志，便于保存实验记录"""
    my_logger = logging.getLogger('my_logger')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    my_logger.addHandler(file_handler)
    my_logger.addHandler(stream_handler)
    my_logger.setLevel(logging.INFO)
    return my_logger


def my_qqplot(y_t, y_p, path, name):
    """
    绘制QQ图
    :param y_t: 测试集真值
    :param y_p: 测试集预测值
    :param project_name: 项目名称 以便保存图片
    :return:
    """
    y_pp = pd.DataFrame()
    y_pp["y_t"] = np.percentile(y_t, range(100))
    y_pp["y_p"] = np.percentile(y_p, range(100))
    plt.figure(figsize=(8, 8))
    # plt.scatter(x='y_t', y='y_p', data=y_pp, label='Actual fit')
    plt.scatter(y_t, y_p, label='Actual fit')
    sns.lineplot(x='y_t', y='y_t', data=y_pp, color='r', label='Line of perfect fit')
    plt.xlabel("test", fontsize=20)
    plt.ylabel("pre", fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=20)
    plt.title("QQ_plot", fontsize=20)
    plt.savefig(f'{path}/{name}.jpg')
    # plt.show()
    plt.clf()


def my_qqplot_bf(y_t, y_p, path, name):
    """
    绘制QQ图
    :param y_t: 测试集真值
    :param y_p: 测试集预测值
    :param project_name: 项目名称 以便保存图片
    :return:
    """
    y_pp = pd.DataFrame()
    y_pp["y_t"] = np.percentile(y_t, range(100))
    y_pp["y_p"] = np.percentile(y_p, range(100))
    plt.figure(figsize=(8, 8))
    plt.scatter(x='y_t', y='y_p', data=y_pp, label='Actual fit')
    # plt.scatter(y_t, y_p, label='Actual fit')
    sns.lineplot(x='y_t', y='y_t', data=y_pp, color='r', label='Line of perfect fit')
    plt.xlabel("test", fontsize=20)
    plt.ylabel("pre", fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=20)
    plt.title("QQ_plot", fontsize=20)
    plt.savefig(f'{path}/{name}_bf.jpg')
    # plt.show()
    plt.clf()


def my_qqplot_yanse(y_t, y_p, path, name):
    """
    绘制QQ图
    :param y_t: 测试集真值
    :param y_p: 测试集预测值
    :param project_name: 项目名称 以便保存图片
    :return:
    """
    y_pp = pd.DataFrame()
    y_pp["y_t"] = np.percentile(y_t, range(100))
    plt.figure(figsize=(8, 8))
    y_pp1 = pd.DataFrame(np.zeros((len(y_t), 2)))
    y_pp1.columns = ["X", "Y"]
    y_pp1["X"] = pd.DataFrame(np.array(y_t).reshape(-1, 1))
    y_pp1["Y"] = pd.DataFrame(np.array(y_p).reshape(-1, 1))
    # 统计每个坐标点的数据个数
    # 绘制散点图
    sns.set_style('whitegrid')
    ax = sns.scatterplot(data=y_pp1, x='X', y='Y', alpha=0.5)

    # 绘制密度图
    sns.kdeplot(data=y_pp1, x='X', y='Y', cmap='Reds', thresh=0.05, fill=True, alpha=0.5, ax=ax)
    sns.lineplot(x='y_t', y='y_t', data=y_pp, color='r', label='Line of perfect fit')
    plt.xlabel("test", fontsize=20)
    plt.ylabel("pre", fontsize=20)
    plt.tick_params(labelsize=15)
    # plt.legend(fontsize=20)
    plt.title("QQ_plot", fontsize=20)
    plt.savefig(f'{path}/{name}_yanse.jpg')
    # plt.show()
    plt.clf()

def get_path(my_path):
    """
    判断是否存在该路径而已
    """
    if not os.path.exists(my_path):
        os.makedirs(my_path)
    return my_path


def log_csv(train, test, config):
    """
    用来自动记录不同训练结果，而不用每次自动手动选择
    :param sota: 预测的具体表型
    :param comfig: 配置参数
    """
    csv_path = os.path.join(config.got_csv, "all.csv")
    if not os.path.exists(csv_path):
        csv_log = pd.DataFrame()
        a1 = []
        a2 = []
        for attr, value in sorted(vars(config).items()):
            a1 += [str(attr)]
            a2 += [str(value)]
        a1 = ["project_name", "model", "dataset", "seed", "train_acc", "text_acc"] + a1
        a2 = [config.project_name,  config.model, config.dataset, config.seed, train, test] + a2
        for j, i in enumerate(a2):
            csv_log.loc[0, j] = i
        csv_log.columns = a1
        csv_log.to_csv(csv_path, index=False)
    else:
        csv_log = pd.read_csv(csv_path, dtype=object)
        csv_log = pd.DataFrame(csv_log)
        b = len(csv_log)
        a1 = []
        a2 = []
        for attr, value in sorted(vars(config).items()):
            a1 += [str(attr)]
            a2 += [str(value)]
        a1 = ["project_name", "model", "dataset", "seed", "train_acc", "text_acc"] + a1
        a2 = [config.project_name,  config.model, config.dataset, config.seed, train, test] + a2
        csv_log.loc[b, :] = a2
        csv_log.columns = a1
        csv_log.to_csv(csv_path, index=False)


