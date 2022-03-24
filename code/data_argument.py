import warnings
warnings.filterwarnings("ignore")
from utils import setup_seed, get_time_dif, view_gpu_info
import pandas as pd
from copy import deepcopy
import math
import random
from tqdm import tqdm
import numpy as np
import json
import re


def denoising(source_file=r'../data/raw_data/train.xlsx', target_file=r'../data/raw_data/train_denoised.xlsx'):
    """
    对原数据 train.xlsx 进行去噪，并保存为 train_denoised.xlsx
    主要进行了以下处理：1 去除多余空格   2 统一'|'   3 字母转换为小写   4 去除约束算子   5 float(nan)转换为''
    @return:
    """
    def process_field(field):
        """
        对字段进行处理： 1 去掉空格   2 统一'|'   3 全部用小写
        @param field
        @return: field
        """
        field = field.replace(' ','').replace('｜', '|')
        field = field.lower()
        return field

    def question_replace(question):
        """
        对问题进行去噪
        :param question:
        :return:
        """
        question = question.replace('二十', '20')
        question = question.replace('三十', '30')
        question = question.replace('四十', '40')
        question = question.replace('五十', '50')
        question = question.replace('六十', '60')
        question = question.replace('七十', '70')
        question = question.replace('八十', '80')
        question = question.replace('九十', '90')
        question = question.replace('一百', '100')
        question = question.replace('十块', '10块')
        question = question.replace('十元', '10元')
        question = question.replace('六块', '6块')
        question = question.replace('一个月', '1个月')
        question = question.replace('2O', '20')
        if '一元一个g' not in question:
            question = question.replace('一元', '1元')
        if 'train' in source_file:
            question = question.replace(' ', '_')
        else:
            question = question.replace(' ', '')
        question = question.lower()
        return question
    raw_data = pd.read_excel(source_file)
    # 训练数据
    if 'train' in source_file:
        raw_data['有效率'] = [1] * len(raw_data)
        for index, row in raw_data.iterrows():
            row['用户问题'] = question_replace(row['用户问题'])
            # TODO 检查是否有效
            # if row['实体'] == '畅享套餐促销优惠':
            #     row['实体'] = '畅享套餐促销优惠活动'
            for index_row in range(len(row)):
                field = str(row.iloc[index_row])
                if field == 'nan':
                    field = ''
                row.iloc[index_row] = process_field(field)
            if not (row['约束算子'] == 'min' or row['约束算子'] == 'max'):
                row['约束算子'] = ''
            raw_data.iloc[index] = row
    # 测试数据
    else:
        length = []
        columns = raw_data.columns
        for column in columns:
            length.append(len(str(raw_data.iloc[1].at[column])))
        max_id = np.argmax(length)
        for index, row in raw_data.iterrows():
            # raw_data.loc[index, 'query'] = question_replace(row['query'])
            raw_data.loc[index, columns[max_id]] = question_replace(row[columns[max_id]])
    # print(raw_data)
    raw_data.to_excel(target_file, index=False)


def read_synonyms():
    """
    读取synonyms.txt 并转换成dict
    @return:
    """
    synonyms_dict = {}
    with open(r'../data/raw_data/synonyms.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            ent, syn_str = line.split()
            syns = syn_str.split('|')
            synonyms_dict[ent] = syns

    return synonyms_dict


def create_argument_data(synonyms_dict):
    """
    对数据进行同义词替换，并保存为新的xlsx
    @param synonyms_dict: 来自synonyms.txt
    @return:
    """
    raw_data = pd.read_excel(r'../data/raw_data/train_denoised.xlsx')
    new_data = []
    for index, row in raw_data.iterrows():
        question = row['用户问题']
        for k, vs in synonyms_dict.items():
            if k in question:
                if '无' in vs:
                    continue
                for v in vs:
                    row_temp = deepcopy(row)
                    row_temp['用户问题'] = question.replace(k, v)
                    new_data.append(row_temp)
    new_data = pd.DataFrame(new_data)
    save_data = pd.concat([raw_data, new_data], ignore_index=True)
    print(save_data)
    save_data.to_excel(r'../data/raw_data/train_syn.xlsx', index=False)


def augment_for_few_data(source_file, target_file, t=20, efficiency=0.8):
    """
    扩充样本量少于t的到t
    :return:
    """
    from nlpcda import Similarword, RandomDeleteChar, CharPositionExchange

    def augment2t_nlpcda(column2id, threshold=t):
        """
        使用nlpcda包来扩充
        """
        from utils import setup_seed
        setup_seed(1)
        a1 = CharPositionExchange(create_num=2, change_rate=0.1, seed=1)
        # a2 = RandomDeleteChar(create_num=2, change_rate=0.1, seed=1)
        a3 = Similarword(create_num=2, change_rate=0.1, seed=1)
        for k, v in tqdm(column2id.items()):
            if len(v) < threshold:
                p = int(math.ceil(threshold / len(v) * 1.0))
                for row_id in v:
                    aug_list = []
                    row = raw_data.loc[row_id]
                    question = str(row['用户问题']).strip()
                    while len(aug_list) < p:
                        aug_list += a1.replace(question)[1:]
                        # aug_list += a2.replace(question)[1:]
                        aug_list += a3.replace(question)[1:]
                        aug_list = list(set(aug_list))
                    for a_q in sorted(aug_list):
                        copy_row = deepcopy(row)
                        copy_row['用户问题'] = a_q
                        target_data.loc[len(target_data)] = copy_row
            # 为并列句去除逗号增强
            if k == '并列句':
                for row_id in v:
                    row = raw_data.loc[row_id]
                    question = str(row['用户问题']).strip()
                    if '，' in question:
                        question = question.replace('，', '')
                        copy_row = deepcopy(row)
                        copy_row['用户问题'] = question
                        target_data.loc[len(target_data)] = copy_row
        target_data['有效率'] = [efficiency] * len(target_data)
    print('开始扩充少量数据...目前阈值为{}'.format(t))
    ans2id, prop2id, entity2id = {}, {}, {}
    raw_data = pd.read_excel(source_file)
    target_data = raw_data.drop(raw_data.index)
    for idx, row in raw_data.iterrows():
        # 答案类型
        if row['答案类型'] not in ans2id:
            ans2id[row['答案类型']] = set()
        ans2id[row['答案类型']].add(idx)
        # 属性名
        prop_list = row['属性名'].split('|')
        for prop in prop_list:
            if prop not in prop2id:
                prop2id[prop] = set()
            prop2id[prop].add(idx)
        # 实体
        if row['实体'] not in entity2id:
            entity2id[row['实体']] = set()
        entity2id[row['实体']].add(idx)
    # 添加
    # augment_to_t(ans2id)
    # augment_to_t(prop2id)
    # augment_to_t(entity2id)
    augment2t_nlpcda(ans2id)
    augment2t_nlpcda(prop2id)
    augment2t_nlpcda(entity2id)
    target_data.to_excel(target_file, index=False)


def augment_for_synonyms(source_file, target_file):
    """
    根据一定的规则，有选择的将同义词替换成原句子
    :param source_file:
    :param target_file:
    :return:
    """
    print('开始进行同义词增强...')
    with open('../data/file/entity_count.json', 'r', encoding='utf-8') as f:
        entity_count = json.load(f)
    with open('../data/file/synonyms.json', 'r', encoding='utf-8') as f:
        synonym_dict = json.load(f)
        entity2synonym = synonym_dict['entity2synonym']
    raw_data = pd.read_excel(source_file)
    target_data = raw_data.drop(raw_data.index)
    random.seed(1)
    for idx, row in tqdm(raw_data.iterrows(), total=len(raw_data)):
        question = row['用户问题']
        entity = row['实体']
        if entity in entity2synonym:
            synonym_list = sorted(entity2synonym[entity] + [entity], key=lambda item: len(item), reverse=True)
            contain_word = ''
            for s in synonym_list:
                if s in question:
                    contain_word = s
                    break
            if contain_word != '':
                synonym_list.remove(contain_word)
                rest_word = synonym_list
                # 根据entity的样本量来进行按概率采样，防止增强的样本量太大
                # 同义词增强，本质上还是需要侧重考虑少样本
                count = entity_count[entity]
                alpha = 1
                # 不需要修改rest_word
                if count <= 30:
                    pass
                elif 30 < count <= 50:
                    rest_word = random.sample(rest_word, int(math.ceil(len(rest_word)*0.5)))
                elif 50 < count <= 100:
                    rest_word = random.sample(rest_word, int(math.ceil(len(rest_word)*0.2)))
                else:
                    if random.random() < 0.5:
                        rest_word = random.sample(rest_word, k=1)
                    else:
                        rest_word = []
                for s in rest_word:
                    copy_row = deepcopy(row)
                    copy_row['用户问题'] = question.replace(contain_word, s)
                    target_data.loc[len(target_data)] = copy_row
    target_data['有效率'] = [0.99] * len(target_data)
    target_data.to_excel(target_file, index=False)


def augment_from_simbert(source_file, target_file,
                         model_file='/data2/hezhenfeng/other_model_files/chinese_simbert_L-6_H-384_A-12',
                         gpu_id=0,
                         start_line=0,
                         end_line=5000,
                         efficiency=0.95):
    """
    利用Simbert生成相似句，取相似度大于95%的句子
    :param source_file: 
    :param target_file: 
    :param model_file:
    :param start_line: 起始行号
    :param end_line: 结束行号
    :param gpu_id:
    :param efficiency:
    :return: 
    """
    from nlpcda import Simbert
    print('加载simbert模型...')
    config = {
        'model_path': model_file,
        'CUDA_VISIBLE_DEVICES': '{}'.format(gpu_id),
        'max_len': 40,
        'seed': 1,
        'device': 'cuda',
        'threshold': efficiency
    }
    simbert = Simbert(config=config)
    raw_data = pd.read_excel(source_file)
    target_data = raw_data.drop(raw_data.index)
    for idx, row in tqdm(raw_data.iterrows(), total=len(raw_data)):
        if start_line <= idx < end_line:
            # 用pandas自带的str类的数据会无法复现结果
            synonyms = simbert.replace(sent=str(row['用户问题']).strip(), create_num=5)
            for synonym, similarity in synonyms:
                if similarity >= config['threshold']:
                    copy_row = deepcopy(row)
                    copy_row['用户问题'] = synonym
                    target_data.loc[len(target_data)] = copy_row
                else:
                    break
        elif idx >= end_line:
            break
    target_data['有效率'] = [efficiency] * len(target_data)
    target_data.to_excel(target_file, index=False)


def function1(args):
    """
    simbert多进程增强的子进程
    :param args:
    :return:
    """
    idx, gpu_id, start, end = args['idx'], args['gpu_id'], args['start'], args['end']
    augment_from_simbert(source_file='../data/raw_data/train_denoised.xlsx',
                         target_file=f'../data/raw_data/train_augment_simbert_{idx}.xlsx',
                         efficiency=0.95,
                         start_line=start,
                         end_line=end,
                         gpu_id=gpu_id)


def run_process():
    """
    simbert多进程增强的子进程
    :return:
    """
    from multiprocessing import Pool
    args = []
    # 初始设置为1
    cpu_worker_num = 1
    span = 5000//cpu_worker_num
    start = 0
    end = span
    for i in range(cpu_worker_num):
        args.append({'idx': i, 'gpu_id': 0, 'start': start, 'end': end})
        start = end
        end += span
        if i == cpu_worker_num-2:
            end = 5000
    with Pool(cpu_worker_num) as p:
        p.map(function1, args)
    print('生成完毕，开始合并结果...')
    df = None
    for i in range(len(args)):
        temp_df = pd.read_excel(f'../data/raw_data/train_augment_simbert_{i}.xlsx')
        if i == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df], ignore_index=True)
    df.to_excel('../data/raw_data/train_augment_simbert.xlsx', index=False)


def multi_process_augment_from_simbert():
    """
    simbert用多进程来完成
    :return:
    """
    import time
    start_time = time.time()
    run_process()
    print('生成时间为：', get_time_dif(start_time))


def augment_for_binary(source_file, target_file):
    """
    同义词增强以及随机去除标点符号
    :return:
    """
    from utils import rm_symbol
    print('开始进行开通方式、条件数据增强...')
    with open('../data/file/synonyms.json', 'r', encoding='utf-8') as f:
        synonym_dict = json.load(f)
        synonym2entity = synonym_dict['synonym2entity']
        for entity in set(synonym2entity.values()):
            synonym2entity[entity] = entity
        synonym_list = sorted(list(synonym2entity.keys()), key=lambda item: [len(item), item], reverse=True)
    raw_data = pd.read_excel(source_file)
    target_data = raw_data.drop(raw_data.index)
    random.seed(1)
    neg_word = ('取消掉', '不需要', '不用', '不想', '不要', '什么时候可以', '可以取消', '可以退订')
    for idx, row in tqdm(raw_data.iterrows(), total=len(raw_data)):
        copy_row = deepcopy(row)
        question_text = str(copy_row['用户问题'])
        p = random.random()
        if '开通' in row['属性名']:
            # 删除这些样本
            if question_text in ('20元20g还可以办理吗', ):
                continue
            # 随机删除逗号
            if p > 0.5 and '，' in row['用户问题']:
                copy_row['用户问题'] = rm_symbol(question_text)
                copy_row['有效率'] = 0.99
                target_data.loc[len(target_data)] = copy_row
            else:
                for synonym in synonym_list:
                    if synonym in question_text:
                        rw = random.choice(synonym_list)
                        if rw != synonym:
                            question_text.replace(synonym, rw)
                        break
                copy_row['有效率'] = 0.99
                copy_row['用户问题'] = question_text
                target_data.loc[len(target_data)] = copy_row
        elif '取消' in row['属性名']:
            flag = False
            for n_w in neg_word:
                if n_w in question_text:
                    flag = True
                    break
            if flag:
                continue
            if '退订' in question_text or '取消' in question_text:
                replace_word = '办理' if p >= 0.5 else '开通'
                question_text = question_text.replace('退订', replace_word)
                question_text = question_text.replace('取消', replace_word)
                copy_row['用户问题'] = question_text
                copy_row['属性名'] = str(copy_row['属性名']).replace('取消', '开通')
                copy_row['有效率'] = 0.9
                target_data.loc[len(target_data)] = copy_row
    target_data.to_excel(target_file, index=False)


def augment_for_ner(source_file, target_file):
    synonyms_dict = read_synonyms()
    df = pd.read_excel(source_file)
    df_syn1 = pd.DataFrame(columns = df.columns)
    df_syn2 = pd.DataFrame(columns = df.columns)
    import collections
    entity_dict = collections.defaultdict(dict)
    # 遍历得到对于每个实体，各个约束出现的次数
    for index, row in df.iterrows():
        if row['约束算子'] == 'min' or row['约束算子'] == 'max':
            continue
        constraint_names = row['约束属性名']
        constraint_values = row['约束属性值']
        constraint_names_list = re.split(r'[｜\|]', str(constraint_names))
        constraint_values_list = re.split(r'[｜\|]', str(constraint_values))
        constraint_names_list = [name.strip() for name in constraint_names_list]
        constraint_values_list = [value.strip() for value in constraint_values_list]
        for i in range(len(constraint_values_list)):
            name = constraint_names_list[i]
            value = constraint_values_list[i]
            if name == '有效期': continue
            if name == 'nan': continue
            if value == '流量套餐': continue
            if name + '_' + value not in entity_dict[row['实体']]:
                entity_dict[row['实体']][name + '_' + value] = []
            entity_dict[row['实体']][name + '_' + value].append(index)
    # 对约束次数少的实体进行同义词替换增强
    for entity, cons in entity_dict.items():
        # 喜马拉雅的ner增强会因错别字无法标注
        if entity == '喜马拉雅流量包': continue
        if entity not in synonyms_dict: continue
        for con, con_list in cons.items():
            if len(con_list) < 3:
                for index in con_list:
                    question = df.iloc[index].at['用户问题']
                    syn_curr = None
                    syn_curr_list = []
                    for syn in synonyms_dict[entity] + [entity]:
                        if syn == '无': continue
                        if syn in question:
                            syn_curr_list.append(syn)
                    syn_curr_list = sorted(syn_curr_list, key=lambda x:len(x))
                    syn_curr = syn_curr_list[-1]
                    for syn in synonyms_dict[entity] + [entity]:
                        if syn == '无': continue
                        new_row = df.iloc[index].copy()
                        new_row.at['用户问题'] = question.replace(syn_curr, syn)
                        df_syn1 = df_syn1.append(new_row, ignore_index=True)
    for index, row in df.iterrows():
        entity = row['实体']
        question = row['用户问题']
        if entity not in synonyms_dict: continue
        syn_curr = None
        syn_curr_list = []
        for syn in synonyms_dict[entity] + [entity]:
            if syn == '无': continue
            if syn in question:
                syn_curr_list.append(syn)
        if len(syn_curr_list) == 0: continue
        syn_curr_list = sorted(syn_curr_list, key=lambda x:len(x))
        syn_curr = syn_curr_list[-1]
        if any(char.isdigit() for char in syn_curr) or '年包' in syn_curr:
            df_syn2 = df_syn2.append(row.copy(), ignore_index=True)
            df_syn2 = df_syn2.append(row.copy(), ignore_index=True)
    df = df.append(df_syn1, ignore_index=True)
    df = df.append(df_syn2, ignore_index=True)
    df.to_excel(target_file, index=False)


if __name__ == '__main__':
    # denoising(source_file='../data/raw_data/test.xlsx', target_file='../data/raw_data/test_denoised.xlsx')
    # synonyms_dict = read_synonyms()
    # create_argument_data(synonyms_dict)
    # setup_seed(1)
    # augment_for_few_data(source_file='../data/raw_data/train_denoised.xlsx',
    #                      target_file='../data/raw_data/train_augment_few_nlpcda.xlsx',
    #                      efficiency=0.8)
    # augment_for_synonyms(source_file='../data/raw_data/train_denoised.xlsx',
    #                      target_file='../data/raw_data/train_augment_synonyms_test2.xlsx')
    # multi_process_augment_from_simbert()
    # augment_from_simbert(source_file='../data/raw_data/train_denoised.xlsx',
    #                      target_file=f'../data/raw_data/train_augment_simbert.xlsx',
    #                      efficiency=0.95,
    #                      start_line=0,
    #                      end_line=5000,
    #                      gpu_id=8)
    augment_for_binary(source_file='../data/raw_data/train_denoised.xlsx',
                       target_file='../data/raw_data/train_augment_binary.xlsx')
