from sklearn.metrics import *
from rdflib import Graph, Namespace
import rdflib
import copy
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import torch
import os
import random
import json
import time
import logging
from datetime import timedelta


class TrainingConfig(object):
    """
    BiLSTM模型的训练参数
    """
    batch_size = 64
    # 学习速率
    lr = 0.001
    epoches = 50
    print_step = 10


class LSTMConfig(object):
    """
    BiLSTM模型中LSTM模块的参数
    """
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数


class BertTrainingConfig(object):
    """
    BiLSTM模型的训练参数
    """
    batch_size = 64
    # 学习速率
    lr = 0.00001
    other_lr = 0.0001
    epoches = 50
    print_step = 5


class BERTLSTMConfig(object):
    """
    BERT+BiLSTM模型中LSTM模块的参数
    """
    emb_size = 768  # 词向量的维数
    hidden_size = 512  # lstm隐向量的维数


def score_evalution(answers, predictions):
    """
    用avr_F1评价预测结果
    @param answers: {'0':'asd|sdf', '1':'qwe|wer' }
    @param predictions: {'0':'asd|sdf', '1':'qwe|wer' }
    @return: avr_F1
    """
    avr_F1 = 0
    for index, answer in answers.items():
        prediction = predictions[index]
        answer_list = answer.split('|')
        answer_set = set()
        for item in answer_list:
            answer_set.add(item)

        prediction_list = prediction.split('|')
        prediction_set = set()
        for item in prediction_list:
            prediction_set.add(item)

        intersection_set = answer_set.intersection(prediction_set)

        A = len(answer_set)
        G = len(prediction_set)
        if G==0 or len(intersection_set) == 0:
            avr_F1 += 0
            continue
        P = len(intersection_set)/(A * 1.0)
        R = len(intersection_set)/(G * 1.0)
        avr_F1 += (2 * P * R)/(P + R)
    avr_F1 /= len(answers)
    return avr_F1

def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    """计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    """
    targets_copy = copy.deepcopy(targets)
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')

    device = crf_scores.device

    # targets_copy:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = targets_copy.size()
    target_size = len(tag2id)

    # mask = 1 - ((targets_copy == pad_id) + (targets_copy == end_id))  # [B, L]
    mask = (targets_copy != pad_id)
    lengths = mask.sum(dim=1)
    targets_copy = indexed(targets_copy, target_size, start_id)

    # # 计算Golden scores方法１
    # import pdb
    # pdb.set_trace()
    targets_copy = targets_copy.masked_select(mask)  # [real_L]

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=targets_copy.unsqueeze(1)).sum()

    # 计算golden_scores方法２：利用pack_padded_sequence函数
    # targets_copy[targets_copy == end_id] = pad_id
    # scores_at_targets = torch.gather(
    #     crf_scores.view(batch_size, max_len, -1), 2, targets_copy.unsqueeze(2)).squeeze(2)
    # scores_at_targets, _ = pack_padded_sequence(
    #     scores_at_targets, lengths-1, batch_first=True
    # )
    # golden_scores = scores_at_targets.sum()

    # 计算all path scores
    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                                      t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_scores) / batch_size
    return loss


def tensorized(batch, maps):
    """
    BiLSTM训练使用
    @param batch:
    @param maps:
    @return:
    """
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def tensorized_bert(batch, maps):
    """
    BiLSTM训练使用
    @param batch:
    @param maps:
    @return:
    """
    PAD = maps.get('[PAD]')
    UNK = maps.get('[UNK]')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets


def get_operator(question):
    """
    得到句子的约束算子，只针对min\max的算子，对算子为‘=’或者空的情况均返回None
    @param question: 用户问题（str）
    @return: 约束算子 和 约束属性
    """
    operater, obj = None, None
    if '最多' in question:
        operater, obj = 'max', '流量'
    elif '最少' in question:
        operater, obj = 'min', '流量'
    elif '最便宜的' in question or '最实惠的' in question:
        operater, obj = 'min', '价格'
    elif '最贵' in question:
        operater, obj = 'max', '价格'
    return operater, obj


def parse_triples_file(file_path):
    """
    解析triples.txt为triples.rdf
    """
    triples = []
    # read raw_triples
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.replace(' ', '')
            line_block = line.split('	')
            t = []
            flag = True
            for block in line_block:
                # 调整三元组格式与训练文件一致
                block = block.replace('档位介绍表', '档位介绍')
                # 三元组改成小写
                block = block.lower()
                if block.find('http://yunxiaomi.com/kbqa') != -1:
                    _ = block.find('_')
                    t.append('http://yunxiaomi.com/'+block[_+1: -1])
                else:
                    block = block[1:-1]
                    if '|' in block:
                        blocks = block.split('|')
                        for b in blocks:
                            flag = False
                            copy_t = copy.deepcopy(t)
                            copy_t.append('http://yunxiaomi.com/'+b)
                            triples.append(copy_t)
                    else:
                        t.append('http://yunxiaomi.com/'+block)
            if flag:
                triples.append(t)
    # transfer to RDF file
    graph = Graph()
    for triple in triples:
        if triple is None:
            print('None')
            continue
        graph.add((rdflib.term.URIRef(u'{}'.format(triple[0])),
                   rdflib.term.URIRef(u'{}'.format(triple[1])),
                   rdflib.term.URIRef(u'{}'.format(triple[2]))))
    graph.serialize('../data/process_data/triples.rdf', format='n3')
    graph.close()


def to_sparql(entity: str, main_property: str, sub_properties: List[Tuple] = None):
    """
    将实体信息等转化成sparql语句
    :param entity: str
    :param main_property: str
    :param sub_properties: List[Tuple[key, value], ...]
    :return: str
    """
    if sub_properties is None:
        sub_properties = []
    prefix = '<http://yunxiaomi.com/{}>'
    # relations第0个元素存放主要关系， 之后存放次要关系
    relations, conditions = [], ''
    # ”档位介绍-取消方式“ 类场景
    if main_property.find('-') != -1:
        relations.append(main_property.split('-')[0])
        relations.append(main_property.split('-')[1])
    # “生效规则” 类场景
    else:
        relations.append(main_property)
    # 填充关系到sparql
    for i, r in enumerate(relations):
        condition = ''
        if i == 0:
            condition = prefix.format(entity) + ' ' + prefix.format(relations[i])
            if len(relations) > 1:
                condition += ' ?instance'
            else:
                condition += ' ?ans'
        elif i == 1:
            condition += '?instance ' + prefix.format(relations[i]) + ' ?ans'
        # else:
        #     condition += '?instance ' + prefix.format(relations[i]) + ' ' + prefix.format(relations[i])
        if len(relations)-1 != i or (len(relations) > 1 and len(sub_properties) > 0):
            condition += '. '
        conditions += condition
    idx = 0
    if len(relations) > 1:
        for key, value in sub_properties:
            condition = '?instance ' + prefix.format(key) + ' ' + prefix.format(value)
            if len(sub_properties) - 1 > idx:
                condition += '. '
            conditions += condition
            idx += 1
    s = """select ?ans where {%s}""" % (conditions, )
    return s


def make_dataset(data_path, target_file, label_file, train=True):
    if isinstance(data_path, list):
        df = pd.DataFrame(pd.read_excel(data_path[0]))
        for i in range(1, len(data_path)):
            df = pd.concat([df, pd.DataFrame(pd.read_excel(data_path[i]))], ignore_index=True)
    else:
        df = pd.DataFrame(pd.read_excel(data_path))
    # # entity_map
    # with open('../data/file/entity_map.json', 'r', encoding='utf-8') as f:
    #     entity_mapping = json.load(f)
    # 标签和id的映射
    ans_label2id = {}
    prop_label2id = {}
    entity_label2id = {}
    # 开通方式、条件
    binary_label2id = {'档位介绍-开通方式': 0, '档位介绍-开通条件': 1}
    # 标注数据xlxs转化成txt
    if train:
        df = df.loc[:, ['用户问题', '答案类型', '属性名', '实体', '答案', '有效率']]
    # 未标注数据xlxs转化成txt
    else:
        length = []
        columns = df.columns
        for column in columns:
            length.append(len(str(df.iloc[1].at[column])))
        max_id = np.argmax(length)
        # df = df.loc[:, ['query']]
        df = df.loc[:, [columns[max_id]]]
    # 转换数据
    with open(target_file, 'w', encoding='utf-8') as f:
        for i in range(len(df)):
            line = df.loc[i]
            line = list(line)
            for idx in range(len(line)):
                # 大写字母改成小写
                line[idx] = str(line[idx]).strip().lower()
            if train:
                # 答案类型
                if line[1] not in ans_label2id:
                    ans_label2id[line[1]] = len(ans_label2id)
                # 属性名
                sub_blocks = line[2].split('|')
                for j, sub_b in enumerate(sub_blocks):
                    # 把这几类统一变成”其他“类
                    # if sub_b in ('适用app', '生效规则', '叠加规则', '封顶规则'):
                    #     sub_b = '其他'
                    #     sub_blocks[j] = sub_b
                    if sub_b not in prop_label2id:
                        prop_label2id[sub_b] = len(prop_label2id)
                # line[2] = '|'.join(list(set(sub_blocks)))
                # # 实体
                # if line[3] in entity_mapping:
                #     line[3] = entity_mapping[line[3]]
                if line[3] not in entity_label2id:
                    entity_label2id[line[3]] = len(entity_label2id)
            f.write('\t'.join(line)+'\n')
    # 整理标签和id的映射
    if train:
        if label_file is not None:
            with open(label_file, 'w', encoding='utf-8') as f:
                json.dump({'ans_type': ans_label2id, 'main_property': prop_label2id, 'entity': entity_label2id,
                           'binary_type': binary_label2id},
                          fp=f,
                          ensure_ascii=False, indent=2)


def make_dataset_for_binary(data_path, target_file):
    """
    为二分类任务制作数据集
    :param data_path:
    :param target_file:
    :return:
    """
    if isinstance(data_path, list):
        df = pd.DataFrame(pd.read_excel(data_path[0]))
        for i in range(1, len(data_path)):
            df = pd.concat([df, pd.DataFrame(pd.read_excel(data_path[i]))], ignore_index=True)
    else:
        df = pd.DataFrame(pd.read_excel(data_path))
    # 标注数据xlxs转化成txt
    df = df.loc[:, ['用户问题', '答案类型', '属性名', '实体', '答案', '有效率']]
    method_count, condition_text = 0, []
    # 转换数据
    with open(target_file, 'w', encoding='utf-8') as f:
        for i in range(len(df)):
            line = df.loc[i]
            if str(line['用户问题']) in ('20元20g还可以办理吗', ):
                continue
            props = str(line['属性名'])
            line = [str(line['用户问题']), '', str(line['有效率'])]
            if '开通方式' in props and '开通条件' not in props:
                line[1] = '档位介绍-开通方式'
                f.write('\t'.join(line)+'\n')
                method_count += 1
            elif '开通方式' not in props and '开通条件' in props:
                line[1] = '档位介绍-开通条件'
                f.write('\t'.join(line)+'\n')
                condition_text.append(line)
        # 中和结果,重采样
        if method_count > len(condition_text):
            for i in range(method_count - len(condition_text)):
                line = random.choice(condition_text)
                f.write('\t'.join(line)+'\n')


def label_to_multi_hot(max_len, label_ids):
    """
    转化成multi-hot编码
    :param max_len: multi-hot编码的最大长度
    :param label_ids:
    :return:
    """
    multi_hot = [0] * max_len
    for idx in label_ids:
        multi_hot[idx] = 1
    return multi_hot


def logits_to_multi_hot(data: torch.Tensor, ans_pred: torch.Tensor, label_hub, threshold=0.5):
    """
    logits转化成multi-hot编码
    1 运用规则： 当并列句中，属性名肯定有两个，而其他句子中属性名只有一个
    :return:
    """
    if data.is_cuda:
        data = data.detach().data.cpu().numpy()
    else:
        data = data.detach().numpy()
    result = []
    for i in range(data.shape[0]):
        temp = [0] * data.shape[1]
        if label_hub.ans_id2label[ans_pred[i].item()] == '并列句':
            # 获取分数最高的两个属性
            max_idx = np.argmax(data[i], axis=-1)
            data[i][max_idx] = -1e5
            temp[max_idx] = 1
            max_idx = np.argmax(data[i], axis=-1)
            temp[max_idx] = 1
        else:
            # 获取分数最高的1个属性
            max_idx = np.argmax(data[i], axis=-1)
            temp[max_idx] = 1
        result.append(temp)
    return np.array(result)


def logits_to_multi_hot_old_version(data: torch.Tensor, threshold=0.5):
    """
    logits转化成multi-hot编码【没有考虑答案类型的旧版本】
    :return:
    """
    if data.is_cuda:
        data = data.detach().data.cpu().numpy()
    else:
        data = data.numpy()
    result = []
    for i in range(data.shape[0]):
        result.append([1 if v >= threshold else 0 for v in list(data[i])])
    return np.array(result)


def remove_stop_words(sentence: str, stopwords: List):
    """
    移除停用词
    :param sentence:
    :param stopwords:
    :return:
    """
    for word in stopwords:
        sentence = sentence.replace(word, '')
    return sentence


def load_model(model_path, map_location=None):
    print('模型加载中...')
    model = torch.load(model_path, map_location=map_location)
    return model


def save_model(model, model_path, model_name, debug='1'):
    # debug模式下，需要删除之前同名的模型
    if debug == '1':
        for file in os.listdir(model_path):
            # 删除同一个模型
            if model_name[:-20] in file:
                os.remove(os.path.join(model_path, file))
    print('模型保存中...')
    torch.save(model, os.path.join(model_path, model_name))


def setup_seed(seed):
    """
    确定随机数
    :param seed: 种子
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。


def split_train_dev(data_size: int, radio=0.8):
    """
    从标注数据中随机划分训练与验证集，返回标注数据中的行号列表
    :param data_size:
    :param radio:
    :return: [[id1, id2, id3...], [idx,... ]]
    """
    line_ids = [i for i in range(data_size)]
    random.shuffle(line_ids)
    train_ids = line_ids[:int(data_size*radio)]
    dev_ids = line_ids[int(data_size*radio):]
    with open('../data/file/train_dev_ids.json', 'w', encoding='utf-8') as f:
        json.dump({'train_ids': train_ids, 'dev_ids': dev_ids}, f, ensure_ascii=False, indent=2)


def split_labeled_data(source_file, train_file, dev_file):
    """
    根据训练与验证的ID，写入文件
    :param source_file:
    :param train_file:
    :param dev_file:
    :return:
    """
    train_dev_ids = json.load(open('../data/file/train_dev_ids.json', 'r', encoding='utf-8'))
    train_dev_ids['dev_ids'] = set(train_dev_ids['dev_ids'])
    train_data, dev_data = [], []
    with open(source_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in train_dev_ids['dev_ids']:
                dev_data.append(line.strip())
            else:
                train_data.append(line.strip())
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    with open(dev_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dev_data))


def get_time_str():
    """
    返回当前时间戳字符串
    :return:
    """
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))


def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_labels(ans_pred: torch.Tensor, prop_pred: np.ndarray, entity_pred: torch.Tensor, label_hub):
    """
    根据标签ID返回标签名称
    :param ans_pred:
    :param prop_pred:
    :param entity_pred:
    :param label_hub:
    :return:
    """
    ans_labels, prop_labels, entity_labels = [], [], []
    for ans_id in ans_pred:
        ans_labels.append(label_hub.ans_id2label[ans_id.item()])
    for entity_id in entity_pred:
        entity_labels.append(label_hub.entity_id2label[entity_id.item()])
    for line in prop_pred:
        temp = []
        for i in range(len(line)):
            if line[i] == 1:
                temp.append(label_hub.prop_id2label[i])
        prop_labels.append(temp)
    return ans_labels, prop_labels, entity_labels


def fetch_error_cases(pred_data: Dict, gold_data: Dict, question: Dict):
    """
    获取预测错误的样本
    :param pred_data:
    :param gold_data:
    :return:
    """
    time_str = get_time_str()
    time_str = time_str.replace(':', '：')
    count = 1
    with open('../data/results/error_case_{}.txt'.format(time_str), 'w', encoding='utf-8') as f:
        for idx, res in pred_data['result'].items():
            ans = gold_data[idx]
            ans = set(ans.split('|'))
            if '' in ans:
                ans.remove('')
            res = set(res.split('|'))
            if ans != res:
                f.write('第{}个错误结果\n'.format(count))
                f.write('问题编号为: {}, 问题为：{}\n'.format(idx, question[idx]))
                f.write('正确答案：{}\n'.format(ans))
                f.write('预测答案：{}\n'.format(res))
                f.write('预测中间结果：{}\n'.format(pred_data['model_result'][idx]))
                f.write('【问题分析】：【】\n')
                f.write('\n')
                count += 1


def k_fold_data(data: List[Dict], k=5, batch_size=32, seed=1, collate_fn='1'):
    """
    data即位DataSet类中的data。将data分成k份，然后组成训练验证集
    :param data:
    :param k:
    :param batch_size:
    :param seed: 随机数种子
    :collate_fn:
    :return:
    """
    print('K折数据划分中...')
    from data import BertDataset, bert_collate_fn, binary_collate_fn
    from torch.utils.data import DataLoader
    temp_data = copy.deepcopy(data)
    random.seed(seed)
    random.shuffle(temp_data)
    block_len = len(temp_data)//k
    data_blocks = [temp_data[i*block_len: (i+1)*block_len] for i in range(k)]
    train_dev_tuples = []
    for i, block in enumerate(data_blocks):
        train, dev = [], block
        for _ in range(k):
            if _ != i:
                train += data_blocks[_]
        train_set = BertDataset(train_file=None, tokenizer=None, label_hub=None, init=False)
        dev_set = BertDataset(train_file=None, tokenizer=None, label_hub=None, init=False)
        train_set.data = train
        dev_set.data = dev
        if collate_fn == '1':
            train_dev_tuples.append((DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn),
                                     DataLoader(dev_set, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)))
        else:
            train_dev_tuples.append(
                (DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=binary_collate_fn),
                 DataLoader(dev_set, batch_size=batch_size, shuffle=True, collate_fn=binary_collate_fn)))
    return train_dev_tuples


def filter_sub_properties(sub_properties: List[Tuple], entity_label: str, ans_label: str):
    """
    过滤 sub_properties
    去掉 实体中流量或价钱被预测成约束的情况 或 去掉重复的约束属性名
    :param sub_properties:
    :param entity_label:
    :param ans_label:
    :return:
    """
    sub_properties_filter = []
    sub_property_set = set()
    for sub_property in sub_properties:
        if sub_property[0] not in sub_property_set:
            sub_property_set.add(sub_property[0])
        # 重复的约束属性名而且答案类型是属性值
        elif ans_label == '属性值':
            continue
        if sub_property[0] == '子业务':
            sub_properties_filter.append(sub_property)
        elif sub_property[1] in entity_label:
            if entity_label == '1元5gb流量券':  # 该实体中 1元 是约束条件
                sub_properties_filter.append(sub_property)
            else:
                continue
        else:
            sub_properties_filter.append(sub_property)
    return sub_properties_filter


def reverse_prop(prop_label: List[str]):
    """
    置换标签，在查找答案为空的时候使用
    :param prop_label:
    :return:
    """
    changed = False
    for i, prop in enumerate(prop_label):
        # 假设以下条件独立
        if '方式' in prop:
            prop_label[i] = prop.replace('方式', '条件')
            changed = True
        elif '条件' in prop:
            prop_label[i] = prop.replace('条件', '方式')
            changed = True
        elif '档位介绍-有效期规则' == prop:
            prop_label[i] = '生效规则'
            changed = True
        elif '生效规则' == prop:
            prop_label[i] = '档位介绍-有效期规则'
            changed = True
    return changed


def reverse_entity(entity_label: str):
    """
    置换属性名，在查找答案为空的时候使用
    :param entity_label:
    :return:
    """
    if entity_label == '嗨购月包':
        return True, '嗨购产品'
    elif entity_label == '嗨购产品':
        return True, '嗨购月包'
    if entity_label == '北京移动plus会员权益卡':
        return True, '北京移动plus会员'
    elif entity_label == '北京移动plus会员':
        return True, '北京移动plus会员权益卡'
    else:
        return False, ''


def rm_symbol(sentence):
    import re
    return re.sub(',|\.|，|。|？|\?|！|!|：', '', sentence)


def judge_cancel(question):
    """
    判断“取消”属性名
    :param question:
    :return:
    """
    import re
    neg_word, pos_word = ['不要', '不想', '取消', '不需要', '不用'], ['办理', '需要', '开通']
    sentence_blocks = re.split(',|\.|：|，|。', question)
    for block in sentence_blocks:
        for n in neg_word:
            if n in block:
                for p in pos_word:
                    if p in block:
                        return True
    return False


def rules_to_judge_prop(question):
    """
    规则判断属性名：适用app、叠加规则、封顶规则、档位介绍-带宽、档位介绍-有效期规则、生效规则
    :param question:
    :return:
    """
    # 适用app：当前只需要判断是否包含“app”和“适用”关键字
    if 'app' in question and '适用' in question:
        return '适用app'
    # 生效规则：判断包含“生效”并带疑问关键字
    if '生效' in question and ('么' in question or '吗' in question or '啥' in question):
        return '生效规则'
    # 叠加规则：包含“可以叠加”关键字。注意与“叠加包”区别
    if '可以叠加' in question and '叠加包' not in question:
        return '叠加规则'
    # 封顶规则：包含“限速”或“上限”关键字，不能包含“解除”、”恢复“关键字
    if ('限速' in question or '上限' in question) and ('解除' not in question and '恢复' not in question):
        return '封顶规则'
    # 有效期规则
    if ('到期' in question or '有效期' in question) and ('办理' not in question and '取消' not in question and '关闭' not in question):
        return '档位介绍-有效期规则'
    # 带宽
    if '通带宽' in question or '网速' in question:
        return '档位介绍-带宽'
    return None


def rules_to_judge_entity(question, predict_entity):
    """
    规则来判断实体
    :param question:
    :param predict_entity
    :return:
    """
    if predict_entity == '畅享套餐':
        if '升档' in question:
            return '新畅享套餐升档优惠活动'
        elif '促销' in question or '78元无限流量套餐' in question:
            return '畅享套餐促销优惠活动'
        elif '新全球通' in question:
            return '新全球通畅享套餐'
        elif '首充活动' in question:
            return '畅享套餐首充活动'
    elif predict_entity == '移动王卡':
        if '惠享合约' in question:
            return '移动王卡惠享合约'
    elif predict_entity == '5g畅享套餐':
        if '合约版' in question:
            return '5g畅享套餐合约版'
    elif predict_entity == '30元5gb包':
        if '半价' in question:
            return '30元5gb半价体验版'
    elif predict_entity == '移动花卡':
        if '新春升级' in question:
            return '移动花卡新春升级版'
    elif predict_entity == '随心看会员':
        if '合约版' in question:
            return '随心看会员合约版'
    elif predict_entity == '北京移动plus会员':
        if '权益卡' in question:
            return '北京移动plus会员权益卡'
    elif predict_entity == '5g智享套餐':
        if '合约版' in question:
            return '5g智享套餐合约版'
        elif '家庭版' in question:
            return '5g智享套餐家庭版'
    elif predict_entity == '全国亲情网':
        if '亲情网免费' in question:
            return '全国亲情网功能费优惠活动'
    elif predict_entity == '精灵卡':
        if '首充' in question and '优惠' in question:
            return '精灵卡首充优惠活动'
    # 合并的样本外的样本
    else:
        if '无忧' in question:
            return '流量无忧包'
    return None


class MyLogger(object):
    def __init__(self, log_file, debug='0'):
        self.ch = logging.StreamHandler()
        self.formatter = logging.Formatter("%(asctime)s - %(message)s")
        self.fh = logging.FileHandler(log_file, mode='w')
        self.logger = logging.getLogger()
        self.debug = debug
        self.init()

    def init(self):
        self.logger.setLevel(logging.INFO)
        # 输出到文件
        self.fh.setLevel(logging.INFO)
        self.fh.setFormatter(self.formatter)
        # 输出到控制台
        self.ch.setLevel(logging.INFO)
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def log(self, message):
        if self.debug == '1':
            self.logger.info(message)


def generate_false_label(predict_file):
    """
    根据预测的结果反向生成同名伪标签xlsx
    :param predict_file
    :return
    """
    result_path = '../data/results/'
    with open(result_path+predict_file, 'r', encoding='utf-8') as f:
        predict_dict = json.load(f)
    predict_results = []
    for i, predict_entity in predict_dict['model_result'].items():
        instance = []
        for key, value in predict_entity.items():
            if key == 'main_property':
                instance.append('|'.join(value))
            elif key == 'sub_properties':
                cons_prop, cons_value = [], []
                for pair in value:
                    cons_prop.append(pair[0])
                    cons_value.append(str(pair[1]))
                instance.append('|'.join(cons_prop))
                instance.append('|'.join(cons_value))
            else:
                if key == 'operator' and value == 'other':
                    instance.append('')
                else:
                    instance.append(value)
        instance.append(predict_dict['result'][i])
        predict_results.append(instance)
    header = ['用户问题', '答案类型', '实体', '属性名', '约束算子', '约束属性名', '约束属性值', '答案']
    pd.DataFrame(predict_results, index=None, columns=header).to_excel(result_path+predict_file[:-5]+'.xlsx', index=False)
    print('预测结果写入xlsx成功')


def view_gpu_info(gpu_id: int):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    print('当前为第{}号显卡，总显存数：{}MB；显存使用数：{}MB；显存剩余数：{}MB'.format(gpu_id, mem_info.total/1024**2, mem_info.used/1024**2, mem_info.free/1024**2))  # 第二块显卡总的显存大小
    return mem_info.total/1024**2, mem_info.used/1024**2, mem_info.free/1024**2


def max_count(data: dict, k=1):
    """
    对于存在"标签"->"个数"的字典data，得到最大数量的标签
    :param data:
    :param k: 前k个统计
    :return:
    """
    max_count_, la = 0, None
    second_count_, la2 = 0, None
    for lab, c in data.items():
        if c > max_count_:
            max_count_ = c
            la = lab
    # 只有在标签个数至少为2时才有第二多的标签，否则直接返回空结果
    if k == 2 and len(data) > 1:
        for lab, c in data.items():
            if c > second_count_ and lab != la:
                second_count_ = c
                la2 = lab
    if k == 1:
        return la, max_count_
    else:
        return la, max_count_, la2, second_count_


def vote_integration(ans_logits: List[torch.Tensor],
                     prop_logits: List[torch.Tensor],
                     entity_logits: List[torch.Tensor],
                     label_hub,
                     batch_size: int):
    """
    模型投票融合
    :param ans_logits:
    :param prop_logits:
    :param entity_logits:
    :param label_hub:
    :param batch_size:
    :return: 最终的标签
    """
    vote_method = 2
    mid_results = {'ans': [], 'prop': [], 'entity': []}
    for i in range(batch_size):
        mid_results['ans'].append({})
        mid_results['prop'].append({})
        mid_results['entity'].append({})
    ans_results, prop_results, entity_results = [], [], []
    for i in range(len(ans_logits)):
        ans_pred = ans_logits[i].data.cpu().argmax(dim=-1)
        prop_pred = logits_to_multi_hot(prop_logits[i], ans_pred, label_hub)
        entity_pred = entity_logits[i].data.cpu().argmax(dim=-1)
        ans_labels, prop_labels, entity_labels = get_labels(ans_pred, prop_pred, entity_pred, label_hub)
        for j in range(batch_size):
            if ans_labels[j] not in mid_results['ans'][j]:
                mid_results['ans'][j][ans_labels[j]] = 0
            mid_results['ans'][j][ans_labels[j]] += 1
            # prop投票1: key是模型预测的直接结果
            if vote_method == 1:
                prop_label = tuple(sorted(prop_labels[j]))
                if prop_label not in mid_results['prop'][j]:
                    mid_results['prop'][j][prop_label] = 0
                mid_results['prop'][j][prop_label] += 1
            # prop投票2: key是模型预测的每个个体标签
            else:
                prop_label = prop_labels[j]
                for lab in prop_label:
                    if lab not in mid_results['prop'][j]:
                        mid_results['prop'][j][lab] = 0
                    mid_results['prop'][j][lab] += 1
            if entity_labels[j] not in mid_results['entity'][j]:
                mid_results['entity'][j][entity_labels[j]] = 0
            mid_results['entity'][j][entity_labels[j]] += 1
    for instance in mid_results['ans']:
        ans_results.append(max_count(instance)[0])
    if vote_method == 1:
        for i, instance in enumerate(mid_results['prop']):
            prop_results.append(max_count(instance)[0])
    else:
        for i, instance in enumerate(mid_results['prop']):
            ans = ans_results[i]
            la, max_count_, la2, second_count_ = max_count(instance, k=2)
            if ans == '并列句':
                if second_count_ <= len(ans_logits)//3:
                    print('-------------------出现了不应该发生的情况-------------------')
                    ans_results[i] = '属性值'
                    prop_results.append([la])
                else:
                    prop_results.append([la, la2])
            else:
                prop_results.append([la])
                if second_count_ > len(ans_logits)//2:
                    prop_results[-1].append(la2)
    for instance in mid_results['entity']:
        entity_results.append(max_count(instance)[0])
    return ans_results, prop_results, entity_results


def average_integration(ans_logits: List[torch.Tensor],
                        prop_logits: List[torch.Tensor],
                        entity_logits: List[torch.Tensor],
                        label_hub):
    """
    模型平均融合
    :param ans_logits:
    :param prop_logits:
    :param entity_logits:
    :param label_hub:
    :return: 最终的标签
    """
    local_ans_logits, local_prop_logits, local_entity_logits = torch.stack(ans_logits).mean(dim=0), torch.stack(prop_logits).mean(dim=0), torch.stack(entity_logits).mean(dim=0)
    ans_pred = local_ans_logits.data.cpu().argmax(dim=-1)
    prop_pred = logits_to_multi_hot(local_prop_logits, ans_pred, label_hub)
    entity_pred = local_entity_logits.data.cpu().argmax(dim=-1)
    ans_labels, prop_labels, entity_labels = get_labels(ans_pred, prop_pred, entity_pred, label_hub)
    return ans_labels, prop_labels, entity_labels


def binary_average_integration(logits: List[torch.Tensor], label_hub):
    """
    二进制模型的平均融合
    :param logits:
    :param label_hub:
    :return:
    """
    local_logits = torch.stack(logits).mean(dim=0)
    pred = local_logits.data.cpu().argmax(dim=-1)
    labels = []
    for label_id in pred:
        labels.append(label_hub.binary_id2label[label_id.item()])
    return labels


def tqdm_with_debug(data, debug=None):
    """
    将tqdm加入debug模式
    :param data:
    :param debug:
    :return:
    """
    from tqdm import tqdm
    if debug == '1' or debug is True:
        if isinstance(data, enumerate):
            temp_data = list(data)
        else:
            temp_data = data
        len_data = len(temp_data)
        ans = tqdm(temp_data, total=len_data)
        return ans
    else:
        return data


# def entity_map():
#     a = {'新畅享套餐升档优惠活动': '畅享套餐', '畅享套餐促销优惠活动': '畅享套餐', '畅享套餐促销优惠': '畅享套餐', '新全球通畅享套餐': '畅享套餐', '畅享套餐首充活动': '畅享套餐',
#          '移动王卡惠享合约': '移动王卡', '5g畅享套餐合约版': '5g畅享套餐', '30元5gb半价体验版': '30元5gb包', '移动花卡新春升级版': '移动花卡', '随心看会员合约版': '随心看会员', '北京移动plus会员权益卡': '北京移动plus会员',
#          '5g智享套餐合约版': '5g智享套餐', '5g智享套餐家庭版': '5g智享套餐',
#          '全国亲情网功能费优惠活动': '全国亲情网', '精灵卡首充优惠活动': '精灵卡'}
#     with open('../data/file/entity_map.json', 'w', encoding='utf-8') as f:
#         json.dump(a, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # parse_triples_file('../data/raw_data/triples.txt')
    # print(to_sparql(entity='视频会员通用流量月包', main_property='档位介绍-上线时间', sub_properties={'价格': '70', '子业务': '优酷会员'}))
    # make_dataset(['../data/raw_data/train_augment_few_nlpcda.xlsx',
    #               '../data/raw_data/train_augment_simbert.xlsx',
    #               '../data/raw_data/train_augment_synonyms.xlsx'],
    #              target_file='../data/dataset/augment3.txt',
    #              label_file=None,
    #              train=True)
    make_dataset_for_binary(['../data/raw_data/train_denoised.xlsx'],
                            target_file='../data/dataset/binary_labeled.txt')
    make_dataset_for_binary(['../data/raw_data/train_augment_few_nlpcda.xlsx',
                             '../data/raw_data/train_augment_simbert.xlsx',
                             '../data/raw_data/train_augment_synonyms.xlsx'],
                            target_file='../data/dataset/binary_augment3.txt')
    # make_dataset(['../data/raw_data/test2_denoised.xlsx'],
    #              target_file='../data/dataset/cls_unlabeled2.txt',
    #              label_file=None,
    #              train=False)
    # split_train_dev(5000)
    # split_labeled_data(source_file='../data/dataset/cls_labeled.txt',
    #                    train_file='../data/dataset/cls_train.txt',
    #                    dev_file='../data/dataset/cls_dev.txt')
    # ------------------获取错误答案详情---------------
    # with open('../data/results/ans_dev.json', 'r', encoding='utf-8') as f:
    #     pred = json.load(f)
    # with open('../data/dataset/cls_dev.txt', 'r', encoding='utf-8') as f:
    #     answer, question = {}, {}
    #     for i, line in enumerate(f):
    #         ans = line.strip().split('\t')[-1]
    #         answer[str(i)] = ans
    #         question[str(i)] = line.strip().split('\t')[0]
    # fetch_error_cases(pred, answer, question)
    # generate_false_label('ensemble_bert_aug2_use_efficiency_2021-07-20-08-29-22_seed1_fi0_gpu2080_be81_0.95411990.pth.json')
    # make_dataset(['../data/raw_data/test2_denoised.xlsx'],
    #              target_file='../data/dataset/cls_unlabeled.txt',
    #              label_file=None,
    #              train=False)
    # entity_map()
