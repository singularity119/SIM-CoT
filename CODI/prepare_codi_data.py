import json
import os

# === 配置路径 ===
# 输入：你服务器上的原始数据路径
INPUT_FILE = "/data2/huangxutao/projects/SIM-CoT/datasets/gsm8k_aug/test.json"

# 输出：train.py 默认寻找的路径 (我们将其标准化存放在 ./data 下)
OUTPUT_DIR = "./datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gsm_test_clean.json")

def main():
    # 1. 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件: {INPUT_FILE}")
        return
    
    print(f"正在读取: {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []

    # 2. 转换逻辑 (Dict of Lists -> List of Dicts)
    # 根据你提供的 test.json 结构，train.json 应该是 {'question': [...], 'cot': [...], 'answer': [...]}
    if isinstance(data, dict) and "question" in data and isinstance(data["question"], list):
        print("检测到数据格式: 列式存储 (Dict of Lists)")
        
        questions = data["question"]
        cots = data.get("cot", [])
        answers = data.get("answer", [])
        
        # 完整性检查
        if not (len(questions) == len(cots) == len(answers)):
            print(f"警告: 数据长度不一致! Q:{len(questions)}, CoT:{len(cots)}, A:{len(answers)}")
            return

        print(f"正在转换 {len(questions)} 条数据...")
        
        for i in range(len(questions)):
            item = {
                "question": questions[i],
                "cot": cots[i],
                "answer": answers[i],
                # 预处理 steps，防止 train.py split 报错 (虽然 icot 数据通常已经是空格分隔的)
                "steps": cots[i].split(" ") if isinstance(cots[i], str) else []
            }
            formatted_data.append(item)
            
    else:
        # 如果格式已经是 List of Dicts，直接复制
        print("检测到数据可能是行式存储或未知格式，尝试直接通过...")
        if isinstance(data, list):
            formatted_data = data
        else:
            print(f"无法识别的数据结构: {type(data)}")
            return

    # 3. 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"💾 正在保存到: {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 完成！共处理 {len(formatted_data)} 条数据。")
    print("-" * 30)
    print("数据样例 (第一条):")
    print(json.dumps(formatted_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()