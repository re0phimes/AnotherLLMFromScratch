"""
清洗训练数据，去除重复pattern
解决数据中大量相同开头导致的过拟合问题
"""
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm


def clean_dataset(
    input_path: str,
    output_path: str,
    max_duplicate_starts: int = 3,
    start_length: int = 50,
    min_text_length: int = 100,
):
    """
    清洗数据集，去除重复开头
    
    Args:
        input_path: 原始数据路径
        output_path: 清洗后数据路径
        max_duplicate_starts: 相同开头最多保留几条
        start_length: 用于判断重复的开头字符数
        min_text_length: 最短文本长度
    """
    
    print("="*80)
    print("数据清洗工具")
    print("="*80)
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return
    
    # 统计信息
    total_count = 0
    kept_count = 0
    filtered_by_length = 0
    filtered_by_duplicate = 0
    
    # 记录已见过的开头
    seen_starts = Counter()
    
    # 第一遍：统计所有开头
    print("\n第一遍：分析数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="统计开头"):
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                if len(text) >= start_length:
                    start = text[:start_length]
                    seen_starts[start] += 1
                total_count += 1
            except:
                continue
    
    print(f"\n分析结果:")
    print(f"  总数据量: {total_count:,}")
    print(f"  唯一开头数: {len(seen_starts):,}")
    
    # 显示最常见的开头
    print(f"\n最常见的开头（前20）:")
    for start, count in seen_starts.most_common(20):
        if count > 1:
            print(f"  '{start}...': {count}次")
    
    # 统计将被过滤的数量
    will_filter = sum(max(0, count - max_duplicate_starts) 
                      for count in seen_starts.values())
    print(f"\n将过滤的重复数据: {will_filter:,} ({will_filter/total_count*100:.2f}%)")
    
    # 第二遍：清洗数据
    print(f"\n第二遍：清洗数据...")
    current_starts = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_count, desc="清洗数据"):
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                
                # 过滤：文本太短
                if len(text) < min_text_length:
                    filtered_by_length += 1
                    continue
                
                # 检查开头重复
                start = text[:start_length] if len(text) >= start_length else text
                
                # 过滤：开头重复次数超限
                if current_starts[start] >= max_duplicate_starts:
                    filtered_by_duplicate += 1
                    continue
                
                # 保留这条数据
                current_starts[start] += 1
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept_count += 1
                
            except Exception as e:
                continue
    
    print(f"\n" + "="*80)
    print("清洗完成！")
    print("="*80)
    
    print(f"\n统计:")
    print(f"  原始数据: {total_count:,}")
    print(f"  保留数据: {kept_count:,} ({kept_count/total_count*100:.2f}%)")
    print(f"  过滤（太短）: {filtered_by_length:,}")
    print(f"  过滤（重复）: {filtered_by_duplicate:,}")
    print(f"\n输出文件: {output_file}")
    
    # 验证输出
    print(f"\n验证输出数据...")
    verify_count = 0
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            verify_count += 1
    
    print(f"  实际写入: {verify_count:,} 条")
    
    if verify_count != kept_count:
        print(f"  ⚠️  警告: 写入数量与预期不符!")
    else:
        print(f"  ✓ 验证通过")


def quick_analysis(data_path: str, sample_size: int = 10000):
    """快速分析数据质量"""
    
    print("="*80)
    print("数据质量分析")
    print("="*80)
    
    texts = []
    starts = []
    
    print(f"\n采样 {sample_size} 条数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                texts.append(text)
                if len(text) >= 50:
                    starts.append(text[:50])
            except:
                continue
    
    # 统计
    total = len(texts)
    unique_starts = len(set(starts))
    duplicate_rate = (len(starts) - unique_starts) / len(starts) if starts else 0
    
    print(f"\n基础统计:")
    print(f"  采样数量: {total}")
    print(f"  唯一开头: {unique_starts}")
    print(f"  重复率: {duplicate_rate*100:.2f}%")
    
    # 长度分布
    lengths = [len(t) for t in texts]
    print(f"\n长度分布:")
    print(f"  平均: {sum(lengths)/len(lengths):.0f} 字符")
    print(f"  最短: {min(lengths)}")
    print(f"  最长: {max(lengths)}")
    
    # 最常见开头
    start_counter = Counter(starts)
    print(f"\n最常见开头（前10）:")
    for start, count in start_counter.most_common(10):
        if count > 1:
            print(f"  '{start}...': {count}次 ({count/len(starts)*100:.2f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="清洗训练数据")
    parser.add_argument(
        '--mode',
        choices=['analyze', 'clean'],
        default='analyze',
        help="模式：analyze=分析数据质量，clean=清洗数据"
    )
    parser.add_argument(
        '--input',
        default='/home/modelenv/chentianxuan/projects/open_source_data_process/data/chinanews_pretrain.jsonl',
        help="输入数据路径"
    )
    parser.add_argument(
        '--output',
        default='/home/modelenv/chentianxuan/projects/open_source_data_process/data/chinanews_pretrain_cleaned.jsonl',
        help="输出数据路径（仅clean模式）"
    )
    parser.add_argument(
        '--max-duplicate',
        type=int,
        default=3,
        help="相同开头最多保留几条"
    )
    parser.add_argument(
        '--start-length',
        type=int,
        default=50,
        help="用于判断重复的开头字符数"
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=100,
        help="最短文本长度"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        quick_analysis(args.input)
    else:
        clean_dataset(
            input_path=args.input,
            output_path=args.output,
            max_duplicate_starts=args.max_duplicate,
            start_length=args.start_length,
            min_text_length=args.min_length,
        )
        
        print("\n" + "="*80)
        print("下一步：更新训练配置")
        print("="*80)
        print("\n修改 configs/train/gpt2_sft_chinanews_fixed.yaml:")
        print(f"  data:")
        print(f"    path: \"{args.output}\"")
