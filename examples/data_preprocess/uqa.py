import argparse
import os
import pandas as pd
import glob

def build_prompt(row):
    return [
        {"role": "system", "content": "你是一位精准的推荐系统专家。请根据用户过去观看一系列视频的互动历史，预测他们接下来最想看的视频和最可能发生的互动行为。"},
        {"role": "user", "content": row['source_text']},
        # {"role": "assistant", "content": f"用户很可能点击视频{row['target_record'][0]['sid']}。"}
    ]

def process_parquet(input_dir, output_dir, data_source="uqa"):
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        def process_row(row):
            # 从 target_record 中提取 sid，并转换为 UQA 奖励函数期望的格式
            target_sid = row['target_record'][0]['sid']
            # 假设 sid 是一个包含三个数字的元组或列表，格式为 (s_a, s_b, s_c)
            # 如果 sid 是字符串，需要解析它
            if isinstance(target_sid, str):
                # 尝试解析字符串格式的 sid
                import re
                numbers = re.findall(r'\d+', target_sid)
                if len(numbers) >= 3:
                    s_a, s_b, s_c = numbers[0], numbers[1], numbers[2]
                else:
                    # 如果无法解析，使用默认值
                    s_a, s_b, s_c = "0", "0", "0"
            elif isinstance(target_sid, (list, tuple)) and len(target_sid) >= 3:
                s_a, s_b, s_c = str(target_sid[0]), str(target_sid[1]), str(target_sid[2])
            else:
                # 如果格式不匹配，使用默认值
                s_a, s_b, s_c = "0", "0", "0"
            
            # 生成符合 UQA 奖励函数期望的格式
            ground_truth = f"<|sid_begin|><s_a_{s_a}><s_b_{s_b}><s_c_{s_c}><|sid_end|>"
            
            return {
                "data_source": data_source,
                "prompt": build_prompt(row),
                "ability": "recommendation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                }
            }
        new_df = pd.DataFrame([process_row(row) for _, row in df.iterrows()])
        output_path = os.path.join(output_dir, os.path.basename(parquet_file))
        new_df.to_parquet(output_path, index=False)
        print(f"已保存到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="输入 parquet 文件目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 parquet 文件目录")
    args = parser.parse_args()

    process_parquet(args.input_dir, args.output_dir)