import subprocess
import os


def get_csv_filenames(directory):
    # 初始化一个空列表存储CSV文件名
    csv_filenames = []
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以.csv结尾
        if filename.endswith(".csv"):
            # 将CSV文件名添加到列表中
            csv_filenames.append(filename)
    return csv_filenames


csv_files = get_csv_filenames("./data")
for namei in csv_files:
    subprocess.run(['python', 'train.py', '--dataset', f"{namei}"])
