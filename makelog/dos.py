import random
import datetime

def generate_random_time(start_time, end_time):
    return start_time + datetime.timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))

def generate_random_ip():
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def generate_dos_log(num_logs):
    logs = []
    start_time = datetime.datetime.strptime("13/Jul/2024:20:00:00", "%d/%b/%Y:%H:%M:%S")
    end_time = datetime.datetime.strptime("13/Jul/2024:20:10:00", "%d/%b/%Y:%H:%M:%S")  # 短时间内的重复请求
    
    for _ in range(num_logs):
        timestamp = generate_random_time(start_time, end_time)
        response_time = round(random.uniform(25.0, 35.0), 3)  # 响应时间在25到35秒之间
        client_ip = generate_random_ip()  # 随机生成IP地址
        
        log_entry = f"{timestamp.strftime('%d/%b/%Y:%H:%M:%S')} +0800 {client_ip} \"\" \"OK\" 413 {response_time} 192 dos"
        logs.append(log_entry)
    
    return logs

# 生成100000条DoS攻击日志
dos_logs = generate_dos_log(100000)

# 将生成的日志写入文件
with open('../log/dos_log.txt', 'w') as f:
    for log in dos_logs:
        f.write(log + '\n')

print("Generated 100,000 DoS log entries.")
