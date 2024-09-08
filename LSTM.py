import pandas as pd
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 读取日志文件并解析数据
def read_log(path):
    dataframe = pd.read_table(path, sep=' ', header=None,
                               names=['timestamp', 'time_zone', 'client_ip', 'cookie', 'status', 'status_code', 'response_time', 'response_size', 'category'],
                               dtype={'timestamp': str, 'time_zone': str, 'client_ip': str, 'cookie': str, 'status': str, 'status_code': int, 'response_time': float, 'response_size': int, 'category': str},
                               na_filter=False)
    dataframe = dataframe.sample(frac=1).values  # 随机打乱数据
    return dataframe

# 将 UTC 时间转换为时间戳
def utc2timestamp(utc_matrix):
    timeStamp = []
    for x in utc_matrix:
        try:
            datetime_str = x[0] + ' ' + x[1]
            timeArray = datetime.datetime.strptime(datetime_str, "%d/%b/%Y:%H:%M:%S %z")
            timeStamp.append([int(time.mktime(timeArray.timetuple()))])
        except ValueError as e:
            print(f"Error converting time: {x} -> {e}")
            timeStamp.append([0])  # 如果发生错误，则返回 0
    return np.array(timeStamp)

# 将 IP 地址转换为二进制格式
def ip2bina(ip_matrix):
    matrix = np.zeros((len(ip_matrix), 32))
    for idx, ip in enumerate(ip_matrix):
        binary = ''.join(format(int(i), '08b') for i in ip.split('.'))
        matrix[idx] = np.array(list(map(int, binary)))
    return matrix

# 将状态码转换为独热编码格式
def status_code2bina(status_code_matrix):
    return np.array([[1, 0, 0, 0] if str(x)[0] == '2' else 
                     [0, 1, 0, 0] if str(x)[0] == '3' else 
                     [0, 0, 1, 0] if str(x)[0] == '4' else 
                     [0, 0, 0, 1] for x in status_code_matrix])

# 将 cookie 转换为二进制格式
def cookie2bina(cookie_matrix):
    return np.array([[1, 0] if x == "" else [0, 1] for x in cookie_matrix])

# 将类别转换为独热编码格式
def category2bina(category_matrix):
    label_encoder = LabelEncoder()
    encoded_categories = label_encoder.fit_transform(category_matrix)
    num_categories = len(label_encoder.classes_)
    return np.eye(num_categories)[encoded_categories]  # 使用独热编码

# 归一化数值矩阵
def normalize(num_matrix):
    amax = np.max(num_matrix, axis=0)
    amin = np.min(num_matrix, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        num_matrix = (num_matrix - amin) / (amax - amin)
    return num_matrix

# 将多个特征矩阵连接为一个输入矩阵
def conn_matx(num_matrix, status_code_matrix, ip_matrix, timestamp_matrix, cookie_matrix):
    return np.hstack((num_matrix, status_code_matrix, ip_matrix, timestamp_matrix, cookie_matrix))

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时刻的输出
        return out

# 训练模型的主函数
def train(csv_file):
    input_infos = read_log(csv_file)
    input_info = input_infos[:, 0:8]
    output_info = input_infos[:, 8]

    # 各种特征的处理
    status_code_matrix = input_info[:, 5].astype(int)
    ip_matrix = input_info[:, 2].astype(str)
    timestamp_matrix = input_info[:, [0, 1]].astype(str)
    category_matrix = input_infos[:, 8].astype(str)
    cookie_matrix = input_info[:, 3].astype(str)
    num_matrix = input_info[:, 6:8].astype(float)

    # 特征转换
    status_code_matrix = status_code2bina(status_code_matrix)
    ip_matrix = ip2bina(ip_matrix)
    timestamp_matrix = normalize(utc2timestamp(timestamp_matrix))
    category_matrix = category2bina(category_matrix)
    cookie_matrix = cookie2bina(cookie_matrix)

    # 拟合 LabelEncoder 并保存
    label_encoder = LabelEncoder()
    label_encoder.fit(output_info)
    joblib.dump(label_encoder, 'label_encoder.pkl')  # 保存 LabelEncoder
    # 输出被编码的类别
    print("Encoded categories:", label_encoder.classes_)


    # 连接所有特征矩阵
    input_matrix = conn_matx(num_matrix, status_code_matrix, ip_matrix, timestamp_matrix, cookie_matrix)

    # 打印输入特征
    print(f"Input feature matrix shape: {input_matrix.shape}")
    print(f"Output feature matrix shape: {output_info.shape}")

    # 分割训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(input_matrix, output_info, test_size=0.1, random_state=42)

    # 打印训练和测试数据的尺寸
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    # 变换输入形状以适应 LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train)
    Y_train_tensor = torch.tensor(label_encoder.transform(Y_train), dtype=torch.long)
    X_test_tensor = torch.FloatTensor(X_test)
    Y_test_tensor = torch.tensor(label_encoder.transform(Y_test), dtype=torch.long)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # 初始化模型、损失函数和优化器
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=64, output_size=len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_loss = float('inf')
    patience = 10
    counter = 0

    # 训练模型
    for epoch in range(10):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(label_encoder.classes_)), targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        # 验证
        val_loss = test_model(model, X_test_tensor, Y_test_tensor, criterion, label_encoder)
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping...")
                break

# 测试模型的函数
def test_model(model, X_test_tensor, Y_test_tensor, criterion, label_encoder):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        test_outputs = model(X_test_tensor)  # 前向传播
        test_loss = criterion(test_outputs.view(-1, len(label_encoder.classes_)), Y_test_tensor)  # 计算测试损失
        print(f"Test Loss: {test_loss.item()}")  # 打印测试损失
        return test_loss.item()  # 返回测试损失

# 主程序入口
if __name__ == '__main__':
    train('./log/train_logs.txt')  # 运行训练函数，指定日志文件路径