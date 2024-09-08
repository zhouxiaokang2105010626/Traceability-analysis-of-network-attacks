from elasticsearch import Elasticsearch
from datetime import datetime

# 连接到 Elasticsearch
es = Elasticsearch([{'host': '114.55.67.147', 'port': 9200}])

# 获取当前时间并格式化为字符串
current_time = datetime.utcnow().strftime("%d/%b/%Y:%H:%M:%S +0000")
iso_time = datetime.utcnow().isoformat() + "Z"

# 要上传的数据列表
documents = [
    {
        "predicted_class": "异常登录",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 尝试使用无效的用户名和密码进行登录，触发了多次失败的登录尝试"
    },
    {
        "predicted_class": "可疑操作",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 执行了未经授权的操作，访问敏感系统文件"
    },
    {
        "predicted_class": "动态蜜罐",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 访问了蜜罐系统中的虚拟服务，触发了警报"
    },
    {
        "predicted_class": "暴力破解",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 进行了一系列的暴力破解尝试，目标是登录界面"
    },
    {
        "predicted_class": "Web后门检测",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 发现了Web后门文件 /var/www/html/backdoor.php"
    },
    {
        "predicted_class": "系统后门检测",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 发现了系统后门，位置 /etc/system_backdoor"
    },
    {
        "predicted_class": "反弹Shell检测",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 发现了反弹 Shell，目标地址为 10.10.10.10"
    },
    {
        "predicted_class": "本地提权",
        "original_message": f"{current_time} 42.49.225.197 \"B01ED3888245229658A0AEDABB75253E\" \"OK\" 200 0.008 1710",
        "timestamp": current_time,
        "client_ip": "42.49.225.197",
        "cookie": "B01ED3888245229658A0AEDABB75253E",
        "status": "OK",
        "status_code": "200",
        "response_time": "0.008",
        "response_size": "1710",
        "public_ip": "42.49.225.197",
        "is_local_ip": "false",
        "city": "Yueyang",
        "region": "Hunan",
        "country": "China",
        "location": {
            "lat": 29.3568,
            "lon": 113.129
        },
        "@timestamp": iso_time,
        "dec": "42.49.225.197 执行了本地提权操作，提升了当前用户权限"
    }
]

if __name__ == '__main__':
    for i, doc in enumerate(documents, 1):
        # 上传数据
        response = es.index(index="predictions_index", id=f"doc{i}", body=doc)
        
        # 输出响应
        print(response)
        
        # 检查上传结果
        if response.get('result') in ['created', 'updated']:
            print(f"数据上传成功: doc{i}")
        else:
            print(f"数据上传失败: doc{i}")
