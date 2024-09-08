import random
import datetime

def random_ip():
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def random_timestamp():
    start_date = datetime.datetime(2024, 7, 1)
    end_date = datetime.datetime(2024, 7, 14)
    random_date = start_date + datetime.timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )
    return random_date.strftime("%d/%b/%Y:%H:%M:%S +0800")

def random_response_time():
    return round(random.uniform(0.01, 1.0), 3)

def generate_log_entry():
    timestamp = random_timestamp()
    ip = random_ip()
    cookie = f'"{random.randint(100000000000000000, 999999999999999999)}"'
    status = "OK"
    status_code = 200
    response_time = random_response_time()
    response_size = random.randint(100, 2000)  # 随机生成响应体大小

    return f'{timestamp} {ip} {cookie} "{status}" {status_code} {response_time} {response_size} gerneral'

def generate_logs(num_entries):
    logs = [generate_log_entry() for _ in range(num_entries)]
    return "\n".join(logs)

if __name__ == "__main__":
    num_logs = 100000  # 生成10万条日志
    logs = generate_logs(num_logs)
    
    # 将日志写入文件
    with open('../log/general_log.txt', 'w') as f:
        f.write(logs)

    print(f"Generated {num_logs} logs and saved to 'generated_logs.txt'")
