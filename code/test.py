"""
@Author: hezf 
@Time: 2021/7/3 19:43 
@desc: 
"""
import warnings
warnings.filterwarnings("ignore")
# from nlpcda import Simbert, Similarword, RandomDeleteChar, Homophone, CharPositionExchange
import time
from multiprocessing import Process
from data_argument import augment_from_simbert
import pandas as pd
import json

# ---simbert---
# config = {
#         'model_path': '/data2/hezhenfeng/other_model_files/chinese_simbert_L-6_H-384_A-12',
#         'CUDA_VISIBLE_DEVICES': '6',
#         'max_len': 40,
#         'seed': 1,
#         'device': 'gpu'
# }
#
# simbert = Simbert(config=config)
# sent_list = ['9元百度专属定向流量包如何取消',
#              '你告诉我7天5g视频会员流量包怎么开通，多少钱',
#                 '您好：怎么开通70爱奇艺',
#                 'plus会员领取的权益可以取消吗',
#                 '通州卡如何取消']
#
# for sent in sent_list:
#     synonyms = simbert.replace(sent=sent, create_num=5)
#     print(synonyms)

# ------------nlpcda 一般增强-------------
# start = time.time()
# sent_list = ['9元百度专属定向流量包如何取消',
#              '你告诉我7天5g视频会员流量包怎么开通，多少钱',
#                 '您好：怎么开通70爱奇艺',
#                 'plus会员领取的权益可以取消吗',
#                 '通州卡如何取消']
#
# smw = CharPositionExchange(create_num=3, change_rate=0.01)
# for sent in sent_list:
#     rs1 = smw.replace(sent)
#     print(rs1)
# end = time.time()
# print(end-start)

# ---------textda---------------
# from utils import setup_seed
# from textda.data_expansion import data_expansion
# import random
#
# if __name__ == '__main__':
#     # setup_seed(1)
#     random.seed(1)
#     print(data_expansion('这是一句测试的句子。'))

# ------------多进程------------

# def function1(id):  # 这里是子进程
#     augment_from_simbert(source_file='../data/raw_data/train_denoised.xlsx',
#                          target_file='../data/raw_data/train_augment_simbert.xlsx')
#
#
# def run_process():  # 这里是主进程
#     from multiprocessing import Process
#     process = [Process(target=function1, args=(1,)),
#                Process(target=function1, args=(2,)), ]
#     [p.start() for p in process]  # 开启了两个进程
#     [p.join() for p in process]   # 等待两个进程依次结束


# ------------标注文件转化成预测样本的格式---------------
def to_predict_file():
    data = []
    with open('../data/dataset/cls_labeled.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip().split('\t')[0])
    with open('../data/dataset/labeled_predict.txt', 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line+'\n')


def xlsx2json():
    data = {}
    df = pd.read_excel('../data/raw_data/train_denoised.xlsx')
    for i in range(len(df)):
        line = df.loc[i]
        data[str(i)] = {
            'question': line['用户问题'],
            'ans_type': line['答案类型'],
            'entity': line['实体'],
            'main_property': line['属性名']}
    with open('../data/results/train_result.json', 'w', encoding='utf-8') as f:
        json.dump({'model_result': data}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    # xlsx2json()
    from tqdm import tqdm
    from utils import judge_cancel
    yes = 0
    # with open('../data/dataset/cls_labeled.txt', 'r', encoding='utf-8') as f:
    #     for line in tqdm(f):
    #         instance = line.strip().split('\t')
    #         if judge_cancel(instance[0]):
    #             if '取消' not in instance[2]:
    #                 print(instance)
    #             else:
    #                 yes += 1
    # print(yes)
    with open('../data/dataset/cls_unlabeled2.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            instance = line.strip().split('\t')
            if judge_cancel(instance[0]):
                print(instance)
