import os
import joblib
import numpy as np
import pandas as pd
import re

data = pd.read_csv(r"./SAM_prediction.csv", header=None).iloc[:, 3]
data1 = pd.DataFrame(data)
data1.columns = ["Gene"]
vocab = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
def decoder(data):
    mydata = np.zeros((len(data), len(data[0].replace("nan", ""))))
    for j in range(len(data)):
        a = np.array(re.findall(r'.{1}', data[j].replace("nan", "")))
        for m, i in enumerate(a):
            a[m] = vocab.get(i)
        mydata[j] = a
    return mydata
data = decoder(data)

def get_out_filenames(directory):
    # 初始化一个空列表存储CSV文件名
    csv_filenames = []
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以.csv结尾_是就pass
        if filename.endswith("csv"):
            # 将CSV文件名添加到列表中
            continue
        csv_filenames.append(filename)
    return csv_filenames


out_files = get_out_filenames("./File-output")
print(out_files)

for namei in out_files:
    model = joblib.load(f"./File-output/{namei}/best_model.pth.tar")
    predictions = model.predict(data)
    predictions = np.array(predictions).reshape(-1, 1)
    namei = namei.split(".")[0]
    data1[f"{namei}"] = pd.DataFrame(predictions)
    print("已完成预测：", namei)
pd.DataFrame(data1).to_csv("predict.csv", index=True)

