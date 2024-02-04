import numpy as np
from scipy.stats import entropy

def eod(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    '''
    对于多元预测的数据集而言, 需要针对每一元进行一个equal_opportunity划分, 比如对于某一种职业, 二元分类目标就是是这种职业/不是这种职业
    所以要求的输入是一个多维的预测概率数组，每一维代表了一个多元类别的预测概率, 如果是二元分类, 那么就只需要传入预测为1的那一维的概率
    对于多元需要先argmax处理一下, 变成分类结果才行
    '''
    # Find the index of the max probability in each row (predicted class)
    predicted_classes = np.argmax(y_pred, axis=1)

    # Create a result matrix where only the predicted class is True for each sample
    result_matrix = np.equal(np.arange(y_pred.shape[1]), predicted_classes[:, None])
    
    _, num_cols = result_matrix.shape
    for col_index in range(num_cols):
        category = result_matrix[:, col_index]
        # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is positive.
        # if num_cols == 1:
        #     category = category[y_gt == 1]
        #     sensitive_attribute = sensitive_attribute[y_gt == 1]
        # else:
        category = category[y_gt == col_index]
        sensitive_positive = sensitive_attribute[y_gt == col_index]

        y_z_1 = category[sensitive_positive == 1] 
        y_z_0 = category[sensitive_positive == 0] 

        # print("y_z_1:", y_z_1)
        # print("y_z_0:", y_z_0)
        # If there are no data points in one of the sensitive attribute groups, return 0.
        if len(y_z_1) == 0 or len(y_z_0) == 0:
            return 0

        # Calculate the difference in true positive rate.
        equality = abs(y_z_1.mean() - y_z_0.mean())
        equality *= 100
        print(f"eod of category {col_index} is: ", equality)

def kld(y_pred, sensitive_attribute) -> float:
    '''
    分类种类是多元的, 敏感类别是2元的
    '''
    def calculate_distribution(pred):
        my_array = np.argmax(pred, axis=1)
        unique_elements, element_counts = np.unique(my_array, return_counts=True)

        n = max(unique_elements)
        element_proportions = np.zeros(n + 1)
        
        # 计算每个类别的概率分布，对于Wikibia就是按照职业划分类别，其他二元的就按照二元分类目标进行划分。
        element_proportions[unique_elements] = element_counts / len(my_array)
        return element_proportions

    # 对于wikitalk而言，sensitive就是找对立的entity
    p1_pred = y_pred[sensitive_attribute == 0]
    p2_pred = y_pred[sensitive_attribute == 1]

    p1_distribution = calculate_distribution(p1_pred)
    p2_distribution = calculate_distribution(p2_pred)

    kld_value = entropy(p1_distribution, p2_distribution)
    return kld_value

def spd(y_pred, sensitive_list):
    '''
    分类种类可以多远, 敏感类别得是2元的
    y_pred是一维输入, 这个一维输入是预测某一类别(如果是二元分类, 那么就是输入为是的那一类概率，如果是多元分类, 就是输入分类为那一类的概率)为阳性类别的概率
    adult: >= 50k 的概率
    acs: >= 50k 的概率
    bias: 分类为这一类的概率
    md_gender: 分类为男性的概率(1)
    wikibias: 分类为bias(1)的概率
    wikitalk: 分类为true(1)的概率
    '''
    # 检查输入的形状和长度
    assert len(y_pred) == len(sensitive_list)

    # 将数据分成两个组，例如，男性和女性
    group_1_indices = np.where(sensitive_list == 1)[0]
    group_2_indices = np.where(sensitive_list == 0)[0]

    # 计算每个组的预测阳性率
    group_1_positive_rate = np.mean(y_pred[group_1_indices] > 0.5)
    group_2_positive_rate = np.mean(y_pred[group_2_indices] > 0.5)

    # 计算 Statistical Parity Difference (SPD)
    spd = abs(group_1_positive_rate - group_2_positive_rate)

    return spd