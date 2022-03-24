"""
@Author: hezf 
@Time: 2021/6/3 14:41 
@desc: 
"""
from torch.utils.data import Dataset
import torch
import copy
import json
from utils import label_to_multi_hot, remove_stop_words
import re
import pandas as pd


class BertDataset(Dataset):
    """
    用于bert的Dataset类
    """
    def __init__(self, train_file, tokenizer, label_hub, init=True):
        super(BertDataset, self).__init__()
        self.train_file = train_file
        self.data = []
        if init:
            self.tokenizer = tokenizer
            self.label_hub = label_hub
            # self.stopwords = []
            self.init()

    def init(self):
        # print('加载停用词表...')
        # with open('../data/file/stopword.txt', 'r', encoding='utf-8') as f:
        #     for line in f:
        #         self.stopwords.append(line.strip())
        print('读取数据...')
        with open(self.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                # blocks: 0:问题；1:答案类型;2:属性名;3:实体;4:答案；5:样本有效率
                blocks = line.strip().split('\t')
                # 单独处理”属性名“
                prop_label_ids = [self.label_hub.prop_label2id[label] for label in blocks[2].split('|')]
                prop_label = label_to_multi_hot(len(self.label_hub.prop_label2id), prop_label_ids)
                self.data.append({'token': self.tokenizer(blocks[0],
                                                          add_special_tokens=True, max_length=100,
                                                          padding='max_length', return_tensors='pt',
                                                          truncation=True),
                                  'ans_label': self.label_hub.ans_label2id[blocks[1]],
                                  'prop_label': prop_label,
                                  'entity_label': self.label_hub.entity_label2id[blocks[3]],
                                  'efficiency': float(blocks[-1])})

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class BinaryDataset(Dataset):
    """
    用于bert的Dataset类
    """
    def __init__(self, train_file, tokenizer, label_hub, init=True):
        super(BinaryDataset, self).__init__()
        self.train_file = train_file
        self.data = []
        if init:
            self.tokenizer = tokenizer
            self.label_hub = label_hub
            # self.stopwords = []
            self.init()

    def init(self):
        # print('加载停用词表...')
        # with open('../data/file/stopword.txt', 'r', encoding='utf-8') as f:
        #     for line in f:
        #         self.stopwords.append(line.strip())
        print('读取数据...')
        with open(self.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                # blocks: 0:问题；1:标签；2:样本有效率
                blocks = line.strip().split('\t')
                # 单独处理”属性名“
                self.data.append({'token': self.tokenizer(blocks[0],
                                                          add_special_tokens=True, max_length=100,
                                                          padding='max_length', return_tensors='pt',
                                                          truncation=True),
                                  'label': self.label_hub.binary_label2id[blocks[1]],
                                  'efficiency': float(blocks[-1])})

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class PredDataset(Dataset):
    """
    用于预测的Dataset类
    """
    def __init__(self, data_path, tokenizer):
        super(PredDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data = []
        self.init()

    def init(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split('\t')
                self.data.append({'token': self.tokenizer(line[0], add_special_tokens=True, max_length=100,
                                                          padding='max_length', return_tensors='pt',
                                                          truncation=True),
                                  'question': line[0]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class PredEnsembleDataset(Dataset):
    """
    用于集成预测的Dataset类
    """
    def __init__(self, data_path):
        super(PredEnsembleDataset, self).__init__()
        self.data_path = data_path
        self.data = []
        self.init()

    def init(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def process_data(data_lists, bieo_dict=None, test=False):
    '''
    由[question] 得到 [['你','好','世','界']] [['O','O','B','E']]
    另外 返回的word_list后面需要加上'<end>'，非test tag_list后面也要加
    @param data_lists: question的list
    @param bieo_dict: 由create_test_BIEO得到的标注字典 {question:['O','O','B','E']}
    @return: question逐字的list 和 标注的list
    '''
    data_word_lists = []
    tag_word_lists = []
    for data_list in data_lists:
        data_word_list = []
        for word in data_list:
            data_word_list.append(word)
        data_word_list.append('<end>')
        data_word_lists.append(data_word_list)
        if bieo_dict is not None:
            tag_word_list = bieo_dict[data_list] + []
            if not test: tag_word_list.append('<end>')
            tag_word_lists.append(tag_word_list)
    if bieo_dict is None:
        return data_word_lists
    return data_word_lists, tag_word_lists


def process_syn(data_list, entities, ans_labels):
    '''
    识别出句子中的实体或者同义词，并将同义词换成实体
    @param data_list:
    @param entities:
    @return:
    '''
    new_data_list = []
    from data_argument import read_synonyms
    syn_dict = read_synonyms()
    for index in range(len(data_list)):
        question = data_list[index]
        ans_label = ans_labels[index]
        entity = entities[index]
        if entity in syn_dict:
            # 排除以下实体，因为同义词中包含有价格、流量、子业务等 #
            if entity in ['视频会员通用流量月包', '1元5gb流量券', '校园卡活动', '新惠享流量券活动', '专属定向流量包',
                          '任我看视频流量包', '快手定向流量包', '5g智享套餐家庭版', '视频会员5gb通用流量7天包',
                          '语音信箱', '全国亲情网', '通话圈', '和家庭分享']:
                new_data_list.append(question)
                continue
            candidate_list = []
            for syn in syn_dict[entity] + [entity]:
                if syn == '无': continue
                if syn == '': continue
                if syn in question:
                    candidate_list.append(syn)
            candidate_list = sorted(candidate_list, key=lambda x:len(x))
            if len(candidate_list) == 0:
                new_data_list.append(question)
                continue
            syn_curr = candidate_list[-1]
            question = question.replace(syn_curr, entity)
            new_data_list.append(question)
        else:
            new_data_list.append(question)
    return new_data_list


def process_postdo(subs, question ,entity):
    if entity == '彩铃':
        if len(subs) == 0:
            if '3元' in question: subs.append(('价格', '3'))
            if '5元' in question: subs.append(('价格', '5'))
    if entity == '承诺低消优惠话费活动':
        if '百度包' in question:
            if ('子业务', '百度包') not in subs:
                subs.append(('子业务', '百度包'))
    if entity == '新惠享流量券活动':
        if '30' in question:
            subs = [('价格', '30')]
            if '40' in question:
                subs.append(('流量', '40'))
        if '20' in question:
            subs = [('价格', '20')]
            if question.count('20') > 1:
                subs.append(('流量', '20'))
    if entity == '神州行5元卡':
        if len(subs) == 0:
            if question.count('5') == 2 or ('5' in question and '30' in question):
                subs.append(('价格', '5'))
    if entity == '承诺低消优惠话费活动':
        if '38' in question:
            subs = [('价格', '38')]
        if '18' in question:
            subs = [('价格', '18')]

    # todo 加入其他后处理
    return subs


def process_postdo_last(question, entity_label, prop_label):
    if '家庭亲情网' in question or '家庭亲情通话' in question or '互打免费' in question:
        entity_label = '全国亲情网'
    elif '家庭亲情号' in question:
        entity_label = '和家庭分享'
        if '成员' in question:
            entity_label = '通话圈'
    elif '家庭亲情' in question:
        entity_label = '和家庭分享'
    # 北京移动plus会员权益卡 和 北京移动plus会员 混淆
    if '权益卡' in question:
        entity_label = '北京移动plus会员权益卡'
    # 30元5gb半价体验版 和 30元5gb 混淆
    if '半价' in question:
        entity_label = '30元5gb半价体验版'
    # 流量无忧包 和 畅享套餐 混淆
    if '无忧' in question:
        entity_label = '流量无忧包'
    if '全国亲情网' in question and '全国亲情网功能费优惠活动' not in question:
        entity_label = '全国亲情网'
    # 和留言服务中的留言服务二字，同时是语音信箱的同义词，容易预测错误
    if '和留言服务' in question:
        entity_label = '和留言'
    # 训练集中所有'帮我开通'都对应'开通方式'
    if '帮我开通' in question:
        if len(prop_label) == 1:
            prop_label = ['档位介绍-开通方式']
    # 训练集中的'怎么取消不了'都对应'取消方式'
    if '怎么取消不了' in question:
        if len(prop_label) == 1:
            prop_label = ['档位介绍-取消方式']
    return entity_label, prop_label

def get_anno_dict(question, anno_list, service_names=None):
    '''
    从bmes标注 得到 实际的 约束属性名 和 约束属性值
    :param question: 用户问题（去掉空格的）
    :param anno_list: bmes的标注 ['O','B','I','E']
    :return: anno_map: {'价钱':['100','20']}
    '''
    type_map = {'PRICE': '价格', 'FLOW': '流量', 'SERVICE': '子业务', 'EXPIRE': '有效期'}
    anno_index = 0
    anno_map = {}
    while anno_index < len(anno_list):
        if anno_list[anno_index] == 'O':
            anno_index += 1
        else:
            if anno_list[anno_index].startswith('S'):
                anno_type = anno_list[anno_index][2:]
                anno_type = type_map[anno_type]
                anno_index += 1
                if anno_type not in anno_map:
                    anno_map[anno_type] = []
                anno_map[anno_type].append(question[anno_index - 1: anno_index])
                continue
            anno_type = anno_list[anno_index][2:]
            anno_type = type_map[anno_type]
            anno_start_index = anno_index
            # B-xxx的次数统计
            B_count = 0
            while not anno_list[anno_index].startswith('E'):
                if anno_list[anno_index] == 'O':
                    anno_index -= 1
                    break
                if anno_index == len(anno_list)-1:
                    break
                if anno_list[anno_index][0] == 'B':
                    B_count += 1
                    if B_count > 1:
                        anno_index -= 1
                        break
                anno_index += 1
            anno_index += 1
            # 对 子业务 和 数字 进行补全
            anno_value = question[anno_start_index: anno_index]
            if service_names != None:
                if anno_type == '子业务':
                    candidate_list = []
                    for service in service_names:
                        if anno_value in service:
                            candidate_list.append(service)
                    # 如果没有符合的实体则筛去该数据
                    if len(candidate_list) == 0:
                        continue
                    drop = True
                    for candidate in candidate_list:
                        if candidate in question:
                            drop = False
                    if drop: continue
                    # 对照句子，找到最符合的对象
                    candidate_list = sorted(candidate_list, key=lambda x:len(x), reverse=True)
                    for candidate in candidate_list:
                        if candidate == '半年包' and anno_value=='年包':break
                        if candidate == '腾讯视频' and anno_value=='腾讯':
                            if '任我看' in question or '24' in question:
                                pass
                            else:
                                break
                        if candidate in question:
                            anno_value = candidate
                            break
                if anno_type == '流量' or anno_type == '价格':
                    l, r = anno_start_index, anno_index
                    while r < len(question) and question[r] == '0':
                        anno_value = anno_value + '0'
                        r += 1
                    while l > 0 and question[l-1].isdigit():
                        anno_value = question[l-1] + anno_value
                        l -= 1
            if anno_type not in anno_map:
                anno_map[anno_type] = []
            # 去除 同一个属性值 出现两次的现象（如果是合约版（比较句出现）、20（价格20流量20）、18（train.xlsx出现），则是正常现象）
            if anno_value in anno_map[anno_type]:
                if anno_value == '合约版' or anno_value == '20' or anno_value == '18':
                    pass
                else: continue
            anno_map[anno_type].append(anno_value)
    # if service_names is not None:
    #     if '和留言' in question and '年包' in question and '月包和留言年包' not in question:
    #         if '子业务' in anno_map:
    #             if '年包' not in anno_map['子业务']:
    #                 anno_map['子业务'].append('年包')
    #         else:
    #             anno_map['子业务'] = ['年包']
    return anno_map


def get_anno_dict_with_pos(question, anno_list, service_names=None):
    '''
    从bmes标注 得到 实际的 约束属性名 和 约束属性值， 以及位置
    :param question: 用户问题（去掉空格的）
    :param anno_list: bmes的标注 ['O','B','I','E']
    :return: anno_map: {'价钱':[['100', 1], ['20', 3]]} 记录值和在句子中出现的位置
    '''
    type_map = {'PRICE': '价格', 'FLOW': '流量', 'SERVICE': '子业务', 'EXPIRE': '有效期'}
    anno_index = 0
    anno_map = {}
    while anno_index < len(anno_list):
        if anno_list[anno_index] == 'O':
            anno_index += 1
        else:
            if anno_list[anno_index].startswith('S'):
                anno_type = anno_list[anno_index][2:]
                anno_type = type_map[anno_type]
                anno_index += 1
                if anno_type not in anno_map:
                    anno_map[anno_type] = []
                anno_map[anno_type].append([question[anno_index - 1: anno_index], anno_index - 1])
                continue
            anno_type = anno_list[anno_index][2:]
            anno_type = type_map[anno_type]
            anno_start_index = anno_index
            # B-xxx的次数统计
            B_count = 0
            while not anno_list[anno_index].startswith('E'):
                if anno_list[anno_index] == 'O':
                    anno_index -= 1
                    break
                if anno_index == len(anno_list)-1:
                    break
                if anno_list[anno_index][0] == 'B':
                    B_count += 1
                    if B_count > 1:
                        anno_index -= 1
                        break
                anno_index += 1
            anno_index += 1
            # 对 子业务 和 数字 进行补全
            anno_value = question[anno_start_index: anno_index]
            pos = anno_start_index
            if service_names != None:
                if anno_type == '子业务':
                    candidate_list = []
                    for service in service_names:
                        if anno_value in service:
                            candidate_list.append(service)
                    # 如果没有符合的实体则筛去该数据
                    if len(candidate_list) == 0:
                        continue
                    drop = True
                    for candidate in candidate_list:
                        if candidate in question:
                            drop = False
                    if drop: continue
                    # 对照句子，找到最符合的对象
                    candidate_list = sorted(candidate_list, key=lambda x:len(x), reverse=True)
                    for candidate in candidate_list:
                        if candidate == '半年包' and anno_value=='年包':break
                        if candidate == '腾讯视频' and anno_value=='腾讯':
                            if '任我看' in question or '24' in question:
                                pass
                            else:
                                break
                        if candidate in question:
                            anno_value = candidate
                            break
                if anno_type == '流量' or anno_type == '价格':
                    l, r = anno_start_index, anno_index
                    while r < len(question) and question[r] == '0':
                        anno_value = anno_value + '0'
                        r += 1
                    while l > 0 and question[l-1].isdigit():
                        anno_value = question[l-1] + anno_value
                        l -= 1
            if anno_type not in anno_map:
                anno_map[anno_type] = []
            # 去除 同一个属性值 出现两次的现象（如果是合约版（比较句出现）、20（价格20流量20）、18（train.xlsx出现），则是正常现象）
            # enhence版 不需要去重
            # if anno_value in anno_map[anno_type]:
            #     if anno_value == '合约版' or anno_value == '20' or anno_value == '18':
            #         pass
            #     else: continue
            anno_map[anno_type].append([anno_value, pos])
    return anno_map


def pred_collate_fn(batch_data):
    """
    用于用于预测的collate函数
    :param batch_data:
    :return:
    """
    input_ids, token_type_ids, attention_mask = [], [], []
    questions = []
    for instance in copy.deepcopy(batch_data):
        questions.append(instance['question'])
        input_ids.append(instance['token']['input_ids'][0].squeeze(0))
        token_type_ids.append(instance['token']['token_type_ids'][0].squeeze(0))
        attention_mask.append(instance['token']['attention_mask'][0].squeeze(0))
    return torch.stack(input_ids), torch.stack(token_type_ids), \
           torch.stack(attention_mask), questions


def bert_collate_fn(batch_data):
    """
    用于BERT训练的collate函数
    :param batch_data:
    :return:
    """
    input_ids, token_type_ids, attention_mask = [], [], []
    ans_labels, prop_labels, entity_labels = [], [], []
    efficiency_list = []
    for instance in copy.deepcopy(batch_data):
        input_ids.append(instance['token']['input_ids'][0].squeeze(0))
        token_type_ids.append(instance['token']['token_type_ids'][0].squeeze(0))
        attention_mask.append(instance['token']['attention_mask'][0].squeeze(0))
        ans_labels.append(instance['ans_label'])
        prop_labels.append(torch.tensor(instance['prop_label']))
        entity_labels.append(instance['entity_label'])
        efficiency_list.append(instance['efficiency'])
    return torch.stack(input_ids), torch.stack(token_type_ids), \
           torch.stack(attention_mask), torch.tensor(ans_labels), \
           torch.stack(prop_labels), torch.tensor(entity_labels), \
           torch.tensor(efficiency_list, dtype=torch.float)


def binary_collate_fn(batch_data):
    """
    用于二类训练的collate函数
    """
    input_ids, token_type_ids, attention_mask = [], [], []
    labels = []
    efficiency_list = []
    for instance in copy.deepcopy(batch_data):
        input_ids.append(instance['token']['input_ids'][0].squeeze(0))
        token_type_ids.append(instance['token']['token_type_ids'][0].squeeze(0))
        attention_mask.append(instance['token']['attention_mask'][0].squeeze(0))
        labels.append(instance['label'])
        efficiency_list.append(instance['efficiency'])
    return torch.stack(input_ids), torch.stack(token_type_ids), \
           torch.stack(attention_mask), torch.tensor(labels), \
           torch.tensor(efficiency_list, dtype=torch.float)


class LabelHub(object):
    """
    分类任务的Label数据中心
    """
    def __init__(self, label_file):
        super(LabelHub, self).__init__()
        self.label_file = label_file
        self.ans_label2id = {}
        self.ans_id2label = {}
        self.prop_label2id = {}
        self.prop_id2label = {}
        self.entity_label2id = {}
        self.entity_id2label = {}
        self.binary_label2id = {}
        self.binary_id2label = {}
        self.load_label()

    def load_label(self):
        with open(self.label_file, 'r', encoding='utf-8') as f:
            label_dict = json.load(f)
        self.ans_label2id = label_dict['ans_type']
        self.prop_label2id = label_dict['main_property']
        self.entity_label2id = label_dict['entity']
        self.binary_label2id = label_dict['binary_type']
        for k, v in self.ans_label2id.items():
            self.ans_id2label[v] = k
        for k, v in self.prop_label2id.items():
            self.prop_id2label[v] = k
        for k, v in self.entity_label2id.items():
            self.entity_id2label[v] = k
        for k, v in self.binary_label2id.items():
            self.binary_id2label[v] = k


def create_test_BIEO(excel_path, test = True):
    """
    由excel生成对应的bieo标注，并保存，同时测试标注方法的准确性
    @param excel_path: 数据路径
    @return:
    """
    def get_bieo(data):
        '''
        得到bmes
        :param data: train_denoised.xlsx的dataframe
        :return: dict ['question': ['O','B','E']]
        '''
        # TODO
        #  标注失败：最便宜最优惠的标注 应该为 价格1；
        #  三十、四十、 十元（先改三十元）、 六块、 一个月
        #  有效期太少了
        type_map = {'价格': 'PRICE', '流量': 'FLOW', '子业务': 'SERVICE'}
        result_dict = {}

        for index, row in data.iterrows():
            question = row['用户问题']
            char_list = ['O' for _ in range(len(question))]

            constraint_names = row['约束属性名']
            constraint_values = row['约束属性值']
            constraint_names_list = re.split(r'[｜\|]', str(constraint_names))
            constraint_values_list = re.split(r'[｜\|]', str(constraint_values))
            constraint_names_list = [name.strip() for name in constraint_names_list]
            constraint_values_list = [value.strip() for value in constraint_values_list]
            question_len = len(question)
            question_index = 0
            # 在句子中标注constraint
            for cons_index in range(len(constraint_values_list)):
                name = constraint_names_list[cons_index]
                if name == '有效期': continue
                value = constraint_values_list[cons_index]
                if value in question[question_index:]:
                    temp_index = question[question_index:].find(value) + question_index
                    if len(value) == 1:
                        char_list[temp_index] = 'S-' + type_map[name]
                        continue
                    else:
                        for temp_i in range(temp_index + 1, temp_index + len(value) - 1):
                            char_list[temp_i] = 'I-' + type_map[name]
                        char_list[temp_index] = 'B-' + type_map[name]
                        char_list[temp_index + len(value) - 1] = 'E-' + type_map[name]
                    question_index = min(temp_index + len(value), question_len)
                elif value in question:
                    temp_index = question.find(value)
                    if len(value) == 1:
                        char_list[temp_index] = 'S-' + type_map[name]
                        continue
                    else:
                        for temp_i in range(temp_index + 1, temp_index + len(value) - 1):
                            char_list[temp_i] = 'I-' + type_map[name]
                        char_list[temp_index] = 'B-' + type_map[name]
                        char_list[temp_index + len(value) - 1] = 'E-' + type_map[name]
            result_dict[question] = char_list

        return result_dict

    def test_bieo(excel_data):
        '''
        用来测试正则化生成的bieo相比于excel中的真值，准确度如何
        :param: excel中读取的dataframe
        :return:
        '''

        annos = get_bieo(excel_data)

        TP, TN, FP = 0, 0, 0

        for index, row in excel_data.iterrows():
            question = row['用户问题']
            # 获取约束的标注
            anno_list = annos[question]
            anno_dict = get_anno_dict(question, anno_list)
            # 获取约束的真值
            constraint_names = row['约束属性名']
            constraint_values = row['约束属性值']
            constraint_names_list = re.split(r'[｜\|]', str(constraint_names))
            constraint_values_list = re.split(r'[｜\|]', str(constraint_values))
            constraint_dict = {}
            for constraint_index in range(len(constraint_names_list)):
                constraint_name = constraint_names_list[constraint_index].strip()
                if constraint_name == '有效期': continue
                constraint_value = constraint_values_list[constraint_index].strip()
                if constraint_name not in constraint_dict:
                    constraint_dict[constraint_name] = []
                constraint_dict[constraint_name].append(constraint_value)
            # 比较约束的真值和标注
            tp = 0
            anno_kv = []
            constraint_kv = []
            for k, vs in anno_dict.items():
                for v in vs:
                    anno_kv.append(k + v)
            for k, vs in constraint_dict.items():
                for v in vs:
                    constraint_kv.append(k + v)
            # 排除 二者均为空 的情况和 比较句 的情况
            if len(anno_kv) == 0 and constraint_kv[0] == 'nannan': continue
            if len(anno_kv) == 0 and (constraint_kv[0] == '价格1' or constraint_kv[0] == '流量1'): continue

            anno_len = len(anno_kv)
            cons_len = len(constraint_kv)
            for kv in constraint_kv:
                if kv in anno_kv:
                    tp += 1
                    anno_kv.remove(kv)
            if tp != cons_len:
                print('-------')
                print(question)
                print('anno: ', anno_kv)
                print('cons: ', constraint_kv)
            TP += tp
            FP += (cons_len - tp)
            TN += (anno_len - tp)
        print('测试bmes结果：' + 'TP: {}  FP: {}  TN:{}  '.format(TP, FP, TN))

    def add_lost_anno(bieo_dict):
        from triples import KnowledgeGraph

        kg = KnowledgeGraph('../data/process_data/triples.rdf')
        df = pd.read_excel('../data/raw_data/train_denoised.xlsx')
        df.fillna('')
        id_list = set()
        ans_list = []
        for iter, row in df.iterrows():
            ans_true = list(set(row['答案'].split('|')))
            question = row['用户问题']
            ans_type = row['答案类型']
            # 只对属性值的句子做处理
            if ans_type != '属性值':
                continue
            entity = row['实体']
            main_property = row['属性名'].split('|')
            # 排除属性中没有'-'的情况，只要'档位介绍-xx'的情况
            if '-' not in main_property[0]:
                continue
            operator = row['约束算子']
            # 排除operator为min或max的情况
            if operator != 'min' and operator != 'max':
                operator == 'other'
            else:
                continue
            sub_properties = {}
            cons_names = str(row['约束属性名']).split('|')
            cons_values = str(row['约束属性值']).split('|')
            if cons_names == ['nan']: cons_names = []
            for index in range(len(cons_names)):
                if cons_names[index] not in sub_properties:
                    sub_properties[cons_names[index]] = []
                sub_properties[cons_names[index]].append(cons_values[index])
            price_ans, flow_ans, service_ans = kg.fetch_wrong_ans(question, ans_type, entity, main_property, operator,
                                                                  [])
            rdf_properties = {}
            rdf_properties['价格'] = price_ans
            rdf_properties['流量'] = flow_ans
            rdf_properties['子业务'] = service_ans
            compare_result = []
            for name, values in rdf_properties.items():
                for value in values:
                    if value in question:
                        if name in sub_properties and value in sub_properties[name]:
                            continue
                        elif name in sub_properties:
                            if value == '年包' and '半年包' in sub_properties[name]:
                                continue
                            if value == '百度' and '百度包' in sub_properties[name]:
                                continue
                        elif value in entity:
                            continue
                        else:
                            compare_result.append(name + '_' + value)
                            id_list.add(iter)
            if compare_result != []:
                ans_list.append(compare_result)

    raw_data = pd.read_excel(excel_path)

    if test:
        test_bieo(raw_data)
    bieo_dict = get_bieo(raw_data)

    with open(r'../data/file/train_bieo.json', 'w') as f:
        json.dump(bieo_dict, f, indent=2, ensure_ascii=False)

    return bieo_dict


def create_test_BIO(excel_path, test = True):
    """
    由excel生成对应的BIO标注，并保存，同时测试标注方法的准确性
    @param excel_path: 数据路径
    @return:
    """
    def get_bio(data):
        '''
        得到bmes
        :param data: train_denoised.xlsx的dataframe
        :return: dict ['question': ['O','B','I']]
        '''
        # TODO
        #  标注失败：最便宜最优惠的标注 应该为 价格1；
        #  三十、四十、 十元（先改三十元）、 六块、 一个月
        #  有效期太少了 可用规则处理
        type_map = {'价格': 'PRICE', '流量': 'FLOW', '子业务': 'SERVICE'}
        result_dict = {}

        for index, row in data.iterrows():
            question = row['用户问题']
            char_list = ['O' for _ in range(len(question))]

            constraint_names = row['约束属性名']
            constraint_values = row['约束属性值']
            constraint_names_list = re.split(r'[｜\|]', str(constraint_names))
            constraint_values_list = re.split(r'[｜\|]', str(constraint_values))
            constraint_names_list = [name.strip() for name in constraint_names_list]
            constraint_values_list = [value.strip() for value in constraint_values_list]
            question_len = len(question)
            question_index = 0
            # 在句子中标注constraint
            for cons_index in range(len(constraint_values_list)):
                name = constraint_names_list[cons_index]
                if name == '有效期':
                    continue
                value = constraint_values_list[cons_index]
                if value in question[question_index:]:
                    temp_index = question[question_index:].find(value) + question_index
                    char_list[temp_index] = 'B-' + type_map[name]
                    for temp_i in range(temp_index + 1, temp_index + len(value)):
                        char_list[temp_i] = 'I-' + type_map[name]
                    question_index = min(temp_index + len(value), question_len)
                elif value in question:
                    temp_index = question.find(value)
                    if char_list[temp_index] == 'O':
                        char_list[temp_index] = 'B-' + type_map[name]
                        for temp_i in range(temp_index + 1, temp_index + len(value)):
                            if char_list[temp_i] == 'O':
                                char_list[temp_i] = 'I-' + type_map[name]
                            else:
                                print('标注冲突："{}"'.format(question))
                                break
                    else:
                        print('标注冲突。"{}"'.format(question))
            result_dict[question] = char_list

        return result_dict

    def test_bio(excel_data):
        '''
        用来测试正则化生成的BIO相比于excel中的真值，准确度如何
        :param: excel中读取的dataframe
        :return:
        '''

        annos = get_bio(excel_data)
        TP, TN, FP = 0, 0, 0
        for index, row in excel_data.iterrows():
            question = row['用户问题']
            # 获取约束的标注
            anno_list = annos[question]
            anno_dict = get_anno_dict(question, anno_list)
            # 获取约束的真值
            constraint_names = row['约束属性名']
            constraint_values = row['约束属性值']
            constraint_names_list = re.split(r'[｜\|]', str(constraint_names))
            constraint_values_list = re.split(r'[｜\|]', str(constraint_values))
            constraint_dict = {}
            for constraint_index in range(len(constraint_names_list)):
                constraint_name = constraint_names_list[constraint_index].strip()
                if constraint_name == '有效期':
                    continue
                constraint_value = constraint_values_list[constraint_index].strip()
                if constraint_name not in constraint_dict:
                    constraint_dict[constraint_name] = []
                constraint_dict[constraint_name].append(constraint_value)
            # 比较约束的真值和标注
            tp = 0
            anno_kv = []
            constraint_kv = []
            for k, vs in anno_dict.items():
                for v in vs:
                    anno_kv.append(k + v)
            for k, vs in constraint_dict.items():
                for v in vs:
                    constraint_kv.append(k + v)
            # 排除 二者均为空 的情况和 比较句 的情况
            if len(anno_kv) == 0 and constraint_kv[0] == 'nannan':
                continue
            if len(anno_kv) == 0 and (constraint_kv[0] == '价格1' or constraint_kv[0] == '流量1'):
                continue

            anno_len = len(anno_kv)
            cons_len = len(constraint_kv)
            for kv in constraint_kv:
                if kv in anno_kv:
                    tp += 1
                    anno_kv.remove(kv)
            if tp != cons_len:
                print('-------')
                print(question)
                print('anno: ', anno_kv)
                print('cons: ', constraint_kv)
            TP += tp
            FP += (cons_len - tp)
            TN += (anno_len - tp)
        print('测试bmes结果：' + 'TP: {}  FP: {}  TN:{}  '.format(TP, FP, TN))

    raw_data = pd.read_excel(excel_path)

    if test:
        test_bio(raw_data)
    bio_dict = get_bio(raw_data)

    with open(r'../data/file/train_bio.json', 'w') as f:
        json.dump(bio_dict, f, ensure_ascii=False, indent=2)
    return bio_dict


if __name__ == '__main__':
    # ----------------自动创建NER标注文件--------------------
    create_test_BIEO(excel_path='../data/raw_data/train_denoised.xlsx')
    create_test_BIO(excel_path='../data/raw_data/train_denoised.xlsx')
