import os

def merge_log_files(output_file, *input_files):
    """合并多个日志文件为一个文件"""
    with open(output_file, 'w') as outfile:
        for filename in input_files:
            try:
                with open(filename, 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write('\n')  # 添加换行符分隔不同文件内容
            except FileNotFoundError:
                print(f"文件 {filename} 未找到，跳过。")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

# 使用示例
if __name__ == '__main__':
    merge_log_files('../log/train_logs.txt', '../log/htpwd_scan_log.txt', '../log/dos_log.txt', '../log/slow_log.txt', '../log/general_log.txt')
    print("日志文件合并完成，输出文件为 'train_logs.txt'")
