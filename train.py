import torch
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import r2_score, accuracy_score
from utils import *
from config import Config
from predata import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import joblib
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
import re

vocab = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
config = Config()
time_start = time.time()
# 创建并实时保存日志
device = torch.device('cuda')
writer = SummaryWriter(config.path)
logger = bulit_logger(f"{config.path}/log.log")
config.print_params(logger.info)
train_x, train_y, test_x, test_y = predata(config, logger)

def decoder(data):
    mydata = np.zeros((len(data), len(data[0].replace("nan", ""))))
    for j in range(len(data)):
        a = np.array(re.findall(r'.{1}', data[j].replace("nan", "")))
        for m, i in enumerate(a):
            a[m] = vocab.get(i)
        mydata[j] = a
    return mydata


train_x, test_x = decoder(train_x), decoder(test_x)
print(train_x.shape)

if config.model == "GBDT":
    model = GradientBoostingRegressor(random_state=123, verbose=1)
    model.fit(train_x, train_y)
elif config.model == "SVM":
    model = SVC(verbose=1)
    model.fit(train_x, train_y)
elif config.model == "MLP":
    model = MLPClassifier(
        hidden_layer_sizes=(256, 64), activation='relu', solver='adam',
        alpha=0.001, max_iter=250, verbose=1, random_state=config.seed)
    model.fit(train_x, train_y)
elif config.model == "RandomForest":
    # esitimators决策树数量
    model = RandomForestRegressor(n_estimators=500, verbose=1)
    model.fit(train_x, train_y)
elif config.model == "XGBOOST":
    cv_params = {'n_estimators': [500, 600, 700]}
    other_params = {'learning_rate': 0.05, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model1 = XGBRegressor(**other_params)
    # 调参
    model = GridSearchCV(model1, cv_params, scoring='r2', cv=5, )
    model.fit(train_x, train_y, verbose=1)
else:
    print("模型应该为GBDT/SVM/MLP/XGBOOST/RandomForest")


joblib.dump(model, f"{config.path}/best_model.pth.tar")
test_predict = model.predict(test_x)
train_predict = model.predict(train_x)
# print(train_predict)
# train_predict = (train_predict > 0.75).astype(int)
# test_predict1 = (test_predict > 0.75).astype(int)

train_accuracy = accuracy_score(train_predict, train_y)
test_accuracy = accuracy_score(test_predict, test_y)
logger.info('train_accuracy:{}'.format(train_accuracy))
logger.info('test_accuracy:{}'.format(test_accuracy))
time_end = time.time()
time_sum = time_end - time_start
logger.info('运行时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
# 保存测试集预测结果
y_t, y_p = np.squeeze(np.array(test_y)), np.squeeze(np.array(test_predict))

save_data = pd.concat([pd.DataFrame(np.expand_dims(y_t, axis=1)), pd.DataFrame(np.expand_dims(y_p, axis=1))], axis=1)
# print(save_data.shape)
# save_data = pd.concat([save_data, pd.DataFrame(np.expand_dims(y_pre, axis=1))], axis=1)
# print(save_data.shape)
save_data.columns = ["y_t", "y_p"]
save_data.to_csv(f"{config.path}/test_tp.csv")
# 绘制QQ图
# my_qqplot(y_t, y_p, config.path, config.name)
# my_qqplot_bf(y_t, y_p, config.path, config.name)
# my_qqplot_yanse(y_t, y_p, config.path, config.name)
log_csv(train_accuracy, test_accuracy, config)
