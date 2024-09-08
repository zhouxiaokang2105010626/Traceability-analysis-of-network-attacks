import re
import time
import numpy as np
import datetime
import torch
import torch.nn as nn
from elasticsearch import Elasticsearch
import joblib
import requests

# 1. 从Elasticsearch获取实时数据
es = Elasticsearch(['http://114.55.67.147:9200'])

def get_real_time_data(index, query, size=10):
    res = es.search(index=index, body=query, size=size)
    data = [hit['_source'] for hit in res['hits']['hits']]
    return data

# 2. 预处理数据
def utc2timestamp(utc_data):
    try:
        timeArray = datetime.datetime.strptime(utc_data, "%d/%b/%Y:%H:%M:%S %z")
        return int(time.mktime(timeArray.timetuple()))
    except ValueError as e:
        print(f"Error converting time: {utc_data} -> {e}")
        return 0

def ip2bina(ip):
    binary = ''.join(format(int(i), '08b') for i in ip.split('.'))
    return np.array(list(map(int, binary)))

def status_code2bina(status_code):
    if str(status_code)[0] == '2':
        return [1, 0, 0, 0]
    elif str(status_code)[0] == '3':
        return [0, 1, 0, 0]
    elif str(status_code)[0] == '4':
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

def cookie2bina(cookie):
    return [1, 0] if cookie == "" else [0, 1]

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) if max_value != min_value else 0

def conn_matx(num_features, status_code_features, ip_features, timestamp_features, cookie_features):
    return np.hstack((num_features, status_code_features, ip_features, timestamp_features, cookie_features))

def parse_message(message):
    pattern = r'(?P<timestamp>\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}) (?P<client_ip>\d+\.\d+\.\d+\.\d+) "(?P<cookie>[\w]*)" "(?P<status>\w*)" (?P<status_code>\d+) (?P<response_time>[\d\.]+) (?P<response_size>\d+)'
    match = re.match(pattern, message)
    if match:
        return match.groupdict()
    return None

# 查询ip相关信息
def get_ip_info(ip):
    url = f'http://ip-api.com/json/{ip}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}

def preprocess_data(log_data):
    processed_data = []
    for single_log_data in log_data:
        message = single_log_data.get('message', '')
        print(message)
        parsed_data = parse_message(message)
        if parsed_data:
            timestamp_matrix = parsed_data['timestamp']
            ip_matrix = parsed_data['client_ip']
            cookie_matrix = parsed_data['cookie']
            status_code_matrix = int(parsed_data['status_code'])
            num_matrix = [
                float(parsed_data['response_time']),
                float(parsed_data['response_size'])
            ]

            status_code_features = status_code2bina(status_code_matrix)
            ip_features = ip2bina(ip_matrix)
            timestamp_features = normalize(utc2timestamp(timestamp_matrix), 0, 2147483647)
            cookie_features = cookie2bina(cookie_matrix)
            num_features = np.array(num_matrix)

            input_features = conn_matx(num_features, status_code_features, ip_features, [timestamp_features], cookie_features)
            input_features = np.reshape(input_features, (1, 1, input_features.shape[0]))

            processed_data.append(input_features)
    
    return processed_data

# 3. 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_model(model_path, input_size, hidden_size, output_size):
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, processed_data, label_encoder):
    predictions = []
    for data in processed_data:
        input_tensor = torch.FloatTensor(data)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).numpy()
            predicted_class_names = label_encoder.inverse_transform(predicted_class)
            predictions.append(predicted_class_names[0])
    return predictions

def upload_predictions_to_elasticsearch(predictions, log_data, index):
    for prediction, log in zip(predictions, log_data):
        message = log.get('message', '')
        public_ip = log.get('public_ip', '')
        is_local_ip = log.get('is_local_ip', '')
        # 获取 IP 信息
        ip_info = get_ip_info(public_ip)
        parsed_data = parse_message(message)
        
        # 获取纬度和经度
        latitude = ip_info.get('lat', None)
        longitude = ip_info.get('lon', None)
        
        # 生成文档
        doc = {
            "predicted_class": prediction,
            "original_message": message,
            "timestamp": parsed_data.get('timestamp', ''),
            "client_ip": parsed_data.get('client_ip', ''),
            "cookie": parsed_data.get('cookie', ''),
            "status": parsed_data.get('status', ''),
            "status_code": parsed_data.get('status_code', ''),
            "response_time": parsed_data.get('response_time', ''),
            "response_size": parsed_data.get('response_size', ''),
            "public_ip": public_ip,
            "is_local_ip": is_local_ip,
            "city": ip_info.get('city', ''),
            "region": ip_info.get('regionName', ''),
            "country": ip_info.get('country', ''),
            "location": {
                "lat": latitude,
                "lon": longitude
            },
            "@timestamp": log.get('@timestamp', ''),
            "dec": f"{public_ip} 发动了 {prediction} 攻击"
        }
        if(latitude != None and longitude != None):
            # 上传到 Elasticsearch
            es.index(index=index, body=doc)

# 4. 使用模型进行实时预测
if __name__ == '__main__':
    model_path = 'best_model.pth'
    input_size = 41  # 输入特征的维度（根据实际情况调整）
    hidden_size = 64
    output_size = 4
    model = load_model(model_path, input_size, hidden_size, output_size)
    label_encoder = joblib.load('label_encoder.pkl')
    last_timestamp = None
    while True:
        query = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gt": last_timestamp
                    }
                }
            }
        }
        
        print(f"Query: {query}")

        real_time_data = get_real_time_data('access_log-*', query, size=10)
        if real_time_data:
            last_timestamp = max(item['@timestamp'] for item in real_time_data)  # 更新最后时间戳
            processed_real_time_data = preprocess_data(real_time_data)
            predictions = predict(model, processed_real_time_data, label_encoder)
            print(predictions)
            upload_predictions_to_elasticsearch(predictions, real_time_data, "predictions_index")
        time.sleep(5)  # 等待一段时间再进行下一次查询
