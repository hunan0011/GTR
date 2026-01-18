import subprocess
import re
import matplotlib.pyplot as plt
import sys
import os

# ================= 配置区域 =================
# 定义要测试的 Checkpoint 编号列表
checkpoints = [1,2,3,4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# 定义要测试的 Beam Size 列表
beam_sizes = [10,20,30,40,50]

# 模型路径模板
model_path_template = "/hdd02/jiangyutao/Code/output/checkpoint-{}.pt"

# Python解释器路径
python_exec = sys.executable 
# ===========================================

# 用于存储结果数据: { ckpt_id: {'beams': [], 'mrr': [], 'recall': []} }
results_data = {ckpt: {'beams': [], 'mrr': [], 'recall': []} for ckpt in checkpoints}

print(f"开始执行自动化评估...")
print(f"总计任务数: {len(checkpoints) * len(beam_sizes)}")
print("-" * 50)

for ckpt in checkpoints:
    current_model_path = model_path_template.format(ckpt)
    
    # 检查模型文件是否存在
    if not os.path.exists(current_model_path):
        print(f"⚠️ 警告: 模型文件未找到 {current_model_path}，可能会报错")

    for beam in beam_sizes:
        print(f"👉 Running: Checkpoint-{ckpt} | Beam={beam}")
        
        # 1. 执行 Inference
        inference_cmd = [
            python_exec, "inference.py",
            "--BEAM_SIZE", str(beam),
            "--MODEL_PATH", current_model_path
        ]
        
        try:
            subprocess.run(inference_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference 失败 (Ckpt {ckpt}, Beam {beam})")
            print(f"错误信息: {e.stderr.decode().strip()}")
            continue

        # 2. 执行 Eval
        eval_cmd = [python_exec, "eval.py"]
        try:
            proc_result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
            output_text = proc_result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ Eval 失败 (Ckpt {ckpt}, Beam {beam})")
            print(f"错误信息: {e.stderr.decode().strip()}")
            continue

        # 3. 解析结果
        mrr_match = re.search(r"MRR@100:\s+([0-9.]+)", output_text)
        recall_match = re.search(r"Recall@100:\s+([0-9.]+)", output_text)

        if mrr_match and recall_match:
            mrr_val = float(mrr_match.group(1))
            recall_val = float(recall_match.group(1))
            
            results_data[ckpt]['beams'].append(beam)
            results_data[ckpt]['mrr'].append(mrr_val)
            results_data[ckpt]['recall'].append(recall_val)
            
            print(f"   ✅ Success: MRR@100={mrr_val:.4f}, Recall@100={recall_val:.4f}")
        else:
            print(f"⚠️ 解析失败: 无法在 eval.py 输出中找到 MRR 或 Recall")

print("-" * 50)

# ==========================================================
#  👇👇👇 [新增部分] 统计每个 Beam 下的最优 Checkpoint 👇👇👇
# ==========================================================
print("【各 Beam Size 最佳结果统计】")
for beam in beam_sizes:
    best_mrr = -1.0
    best_mrr_ckpt = -1
    
    best_recall = -1.0
    best_recall_ckpt = -1
    
    for ckpt in checkpoints:
        data = results_data[ckpt]
        if beam in data['beams']:
            idx = data['beams'].index(beam)
            
            # 比较 MRR
            if data['mrr'][idx] > best_mrr:
                best_mrr = data['mrr'][idx]
                best_mrr_ckpt = ckpt
            
            # 比较 Recall
            if data['recall'][idx] > best_recall:
                best_recall = data['recall'][idx]
                best_recall_ckpt = ckpt
    
    print(f"Beam Size {beam}:")
    print(f"  Best MRR   : {best_mrr:.4f} (from Checkpoint {best_mrr_ckpt})")
    print(f"  Best Recall: {best_recall:.4f} (from Checkpoint {best_recall_ckpt})")
    print("-" * 30)
# ==========================================================


print("所有任务完成，正在绘图...")

# ================= 绘图逻辑 (保持原样) =================
def plot_metric(metric_name, save_filename):
    plt.figure(figsize=(10, 6))
    
    # 为每个 Checkpoint 画一条线
    for ckpt in checkpoints:
        data = results_data[ckpt]
        if not data['beams']:
            continue
            
        # 提取 y轴数据 (mrr 或 recall)
        y_values = data['mrr'] if metric_name == 'MRR' else data['recall']
        
        plt.plot(data['beams'], y_values, marker='o', linewidth=2, label=f'Checkpoint {ckpt}')

    plt.title(f'{metric_name}@100 vs Beam Size', fontsize=14)
    plt.xlabel('Beam Size', fontsize=12)
    plt.ylabel(f'{metric_name}@100', fontsize=12)
    plt.xticks(beam_sizes) # 强制 x 轴刻度显示为 10, 20...
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.savefig(save_filename, dpi=300)
    print(f"图片已保存: {save_filename}")
    plt.close()

# 画两张图
try:
    plot_metric('Recall', 'recall_100_curve2.png')
    plot_metric('MRR', 'mrr_100_curve2.png')
    print("绘图完成！请查看当前目录下的 .png 图片。")
except Exception as e:
    print(f"绘图时出错: {e}")
    print("但是数据已经跑完了，你可以手动打印 results_data 查看结果。")