import random
import datetime

def generate_random_time(start_time, end_time):
    return start_time + datetime.timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))

def generate_random_ip():
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def generate_slow_log(num_logs):
    logs = []
    start_time = datetime.datetime.strptime("01/Jul/2024:00:00:00", "%d/%b/%Y:%H:%M:%S")
    end_time = datetime.datetime.strptime("15/Jul/2024:23:59:59", "%d/%b/%Y:%H:%M:%S")
    
    for _ in range(num_logs):
        timestamp = generate_random_time(start_time, end_time)
        response_time = round(random.uniform(55, 65), 3)  # 响应时间在55到65秒之间
        client_ip = generate_random_ip()  # 随机生成IP地址
        
        log_entry = f"{timestamp.strftime('%d/%b/%Y:%H:%M:%S')} +0800 {client_ip} \"\" \"\" 408 {response_time} 0 slowlinks"
        logs.append(log_entry)
    
    return logs

# 生成100000条慢链接日志
slow_logs = generate_slow_log(100000)

# 将生成的日志写入文件
with open('../log/slow_log.txt', 'w') as f:
    for log in slow_logs:
        f.write(log + '\n')

print("Generated 100,000 slow log entries.")
