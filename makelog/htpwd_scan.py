import random
import datetime

def generate_random_time(start_time, end_time):
    return start_time + datetime.timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))

def generate_random_ip():
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def generate_htpwd_scan_log(num_logs):
    logs = []
    start_time = datetime.datetime.strptime("13/Jul/2024:20:00:00", "%d/%b/%Y:%H:%M:%S")
    end_time = datetime.datetime.strptime("13/Jul/2024:20:10:00", "%d/%b/%Y:%H:%M:%S")  # 在短时间内重复请求

    for _ in range(num_logs):
        timestamp = generate_random_time(start_time, end_time)
        response_time = round(random.uniform(0.002, 1.009), 3)  # 响应时间在0.002到1.009之间
        client_ip = generate_random_ip()  # 随机生成IP地址

        log_entry = f"{timestamp.strftime('%d/%b/%Y:%H:%M:%S')} +0800 {client_ip} \"\" \"OK\" 302 {response_time} 0 Hitlibrary"
        logs.append(log_entry)

    return logs

# 生成100000条htpwdScan攻击日志
htpwd_scan_logs = generate_htpwd_scan_log(100000)

# 将生成的日志写入文件
with open('../log/htpwd_scan_log.txt', 'w') as f:
    for log in htpwd_scan_logs:
        f.write(log + '\n')

print("Generated 100,000 htpwdScan log entries.")
