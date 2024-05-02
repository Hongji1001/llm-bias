import os
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 替换为你的JSON文件的实际路径
file_path = '../outputs/completions/llama_wikitext_2k.json.json'
output_directory = '../outputs/figures'
os.makedirs(output_directory, exist_ok=True)

# 读取JSON文件
df = pd.read_json(file_path)
# 根据文本的词数进行分段统计
df['WordCount'] = df['completions'].apply(lambda x: len(x.split()))
df['WordCountSegment'] = pd.cut(df['WordCount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], include_lowest=True, right=False)

# 定义一个绘制直方图并添加文本标签的函数
def plot_score_distribution_with_labels(sub_df, score_column, title, filename):
    # 计算每个分段的平均得分和数量
    averages = sub_df.groupby('WordCountSegment').agg({score_column: ['mean', 'count']}).reset_index()
    averages.columns = ['WordCountSegment', 'AverageScore', 'Count']  # 重新命名多级列

    fig, ax = plt.subplots()
    bars = ax.bar(averages['WordCountSegment'].astype(str), averages['AverageScore'], color='skyblue')

    # 在每个条形上添加文本标签
    for bar, avg_score, count in zip(bars, averages['AverageScore'], averages['Count']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{avg_score:.2f}\nN={count}',
                ha='center', va='bottom')

    ax.set_xlabel('Word Count Segment')
    ax.set_ylabel(f'Average {score_column.title()} Score')
    ax.set_title(title)
    ax.set_xticklabels(averages['WordCountSegment'].astype(str), rotation=45)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_directory, filename))
    plt.close()


toxicity_evaluator = evaluate.load("toxicity", module_type="measurement")
cat_result = toxicity_evaluator.compute(predictions=df['completions'].to_list())
toxicity_scores = cat_result["toxicity"]
df['toxicity'] = toxicity_scores
regard_evaluator = evaluate.load("regard", module_type="measurement")
result = regard_evaluator.compute(data=df['completions'].to_list())
negative_scores = [d[0]['score'] for d in result['regard'] for l in d if l['label'] == 'negative']
df['regard'] = negative_scores

# # 示例：为每个domain的toxicity和regard生成图表
# for domain, group in df.groupby('domain'):
#     plot_score_distribution_with_labels(group, 'toxicity', f'{domain} - Toxicity', f"{domain}_toxicity.png")
#     plot_score_distribution_with_labels(group, 'regard', f'{domain} - Regard', f"{domain}_regard.png")

# 为所有domain生成图表
plot_score_distribution_with_labels(df, 'toxicity', "All Domains - Toxicity", "wikitext_all_domains_toxicity.png")
plot_score_distribution_with_labels(df, 'regard', "All Domains - Regard", "wikitext_all_domains_regard.png")