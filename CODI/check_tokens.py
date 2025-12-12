from transformers import AutoTokenizer

# 修改为你服务器上的模型路径
#MODEL_PATH = "/mnt/shared-storage-user/mllm/weixilin/Llama-3.2-1B-Instruct"
MODEL_PATH = "/data2/huangxutao/projects/SIM-CoT/models/llama3.2_1b_instruct/origin_model"

def check_tokens():
    print(f"正在加载 Tokenizer: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 测试 SIM-CoT 数据中常见的关键符号
    # 注意：Llama-3 的 Tokenizer 可能会把 "<<" 拆分成两个 "<" 或者作为一个整体
    test_strings = ["<<", ">>", "\n", "step"]
    
    print("-" * 40)
    print("关键符号 Token ID 测试：")
    print("-" * 40)
    
    for s in test_strings:
        # encode 可能会自动加上 begin_of_text token，我们需要去掉它
        ids = tokenizer.encode(s, add_special_tokens=False)
        print(f"符号: '{s}'  \t-> IDs: {ids}")

    print("-" * 40)
    print("请根据上面的输出修改 src/model.py 中的 get_steps 函数。")
    print("例如，如果 '<<' 的 ID 是 [2345]，就把 start_ids 改为 (2345, )")

if __name__ == "__main__":
    check_tokens()