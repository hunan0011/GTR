'''
Author: “Mia” “welshcorgi@foxmail.com”
Date: 2026-01-25 15:57:39
LastEditors: “Mia” “welshcorgi@foxmail.com”
LastEditTime: 2026-02-11 17:59:01
FilePath: /jiangyutao/GTR/run.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import subprocess
import sys
import time

# 这里列出你想运行的脚本路径
scripts_to_run = [
    "construct_tree_gmm.py",
    "train.py",
    "inference.py"
]

def run_scripts_sequentially(scripts):
    for script in scripts:
        print(f"----- 开始运行: {script} -----")
        start_time = time.time()
        
        try:
            # 使用当前 Python解释器 (sys.executable) 运行脚本
            # check=True 表示如果脚本报错，会抛出异常停止后续运行
            result = subprocess.run([sys.executable, script], check=True)
            
            end_time = time.time()
            print(f"----- {script} 运行完成，耗时: {end_time - start_time:.2f}秒 -----\n")
            
        except subprocess.CalledProcessError as e:
            print(f"!!!!! {script} 运行出错，退出代码: {e.returncode} !!!!!")
            # 如果你想出错后继续运行下一个，可以在这里写 pass
            break 
        except Exception as e:
            print(f"!!!!! 发生未知错误: {e} !!!!!")
            break

if __name__ == "__main__":
    run_scripts_sequentially(scripts_to_run)