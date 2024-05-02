import yaml
import re

# 假设原始YAML文件路径是 'original_jobs.yaml'
original_file_path = 'profession.yaml'
# 目标文件路径
new_file_path = 'occupation.yaml'

# 读取原始YAML文件
with open(original_file_path, 'r') as file:
    original_data = yaml.safe_load(file)

# 新的分类逻辑
new_classification = {
    'Professional_and_Management_Services': [],
    'Technology_and_Engineering': [],
    'Social_Services_and_Creative_Industries': []
}

# 定义每个新分类下包含的原始分类
classification_mapping = {
    'Professional_and_Management_Services': [
        'Administration', 'Business and finance', 'Law and legal', 'Managerial', 'Retail and sales',
        'Government services'
    ],
    'Technology_and_Engineering': [
        'Computing, technology and digital', 'Engineering and maintenance', 'Construction and trades',
        'Science and research', 'Manufacturing'
    ],
    'Social_Services_and_Creative_Industries': [
        'Animal care', 'Beauty and wellbeing', 'Creative and media', 'Healthcare', 'Social care', 'Sports and leisure',
        'Teaching and education', 'Home services', 'Hospitality and food', 'Emergency and uniform services',
        'Environment and land'
    ]
}

# 根据映射填充新分类
for new_cat, old_cats in classification_mapping.items():
    for old_cat in old_cats:
        if old_cat in original_data:    # 确保原始数据中存在这个类别
            new_classification[new_cat].extend(original_data[old_cat])

yaml_str = yaml.dump(new_classification, default_flow_style=False, indent=4)

# 使用正则表达式替换每个列表项前的空格为Tab
yaml_str_with_tabs = re.sub(r"\n(\s+)-", r"\n\1\t-", yaml_str)

# 写入文件
with open(new_file_path, 'w') as file:
    file.write(yaml_str_with_tabs)

print('Reclassification completed and saved to', new_file_path)
