import pandas as pd
import re
import json
import os
import random
import time
import torch
from transformers import BertModel, BertTokenizer

from model import Metrics, BILSTM_CRF_Model, BERT_BILSTM_CRF_Model
from data import *
from utils import setup_seed, load_model
from statistic import get_all_service, statistic_wrong_cons_bieo
import argparse


def train(excel_path, model_name, tag_version='bio'):

    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e == '<end>': continue
                if e not in maps:
                    maps[e] = len(maps)
        maps['<unk>'] = len(maps)
        maps['<pad>'] = len(maps)
        maps['<start>'] = len(maps)
        maps['<end>'] = len(maps)
        return maps

    # data
    print("读取数据...")
    with open(r'../data/file/train_{}.json'.format(tag_version)) as f:
            train_dict = json.load(f)
    # with open(r'../data/file/train_dev_ids.json') as f:
    #     id_dict = json.load(f)
    raw_data = pd.read_excel(excel_path)
    question_list = raw_data['用户问题'].tolist()
    question_list = [question.replace(' ', '') for question in question_list]
    for i in range(len(question_list)):
        question = question_list[i]
        question = question.replace('三十', '30')
        question = question.replace('四十', '40')
        question = question.replace('十块', '10块')
        question = question.replace('六块', '6块')
        question = question.replace('一个月', '1个月')
        question = question.replace('2O', '20')
        question_list[i] = question
    # 按照划分好的id划分数据
    # train_list = [question_list[i] for i in id_dict['train_ids']]
    # dev_list = [question_list[i] for i in id_dict['dev_ids']]
    # test_list = [question_list[i] for i in id_dict['dev_ids']]
    # 随机划分数据
    random.shuffle(question_list)
    train_list = question_list[:int(0.8*len(question_list))]
    dev_list = question_list[int(0.8*len(question_list)): int(0.9*len(question_list))]
    test_list = question_list[int(0.9*len(question_list)):]

    train_word_lists, train_tag_lists = process_data(train_list, train_dict)
    dev_word_lists, dev_tag_lists = process_data(dev_list, train_dict)
    test_word_lists, test_tag_lists = process_data(test_list, train_dict, test=True)
    # 生成word2id 和 tag2id 并保存
    word2id = build_map(train_word_lists)
    tag2id = build_map(train_tag_lists)
    with open(r'../data/file/word2id.json', 'w') as f: json.dump(word2id, f)
    with open(r'../data/file/tag2id_{}.json'.format(tag_version), 'w') as f: json.dump(tag2id, f)
    # train
    print("正在训练评估Bi-LSTM+CRF模型...")
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstmcrf_model = BILSTM_CRF_Model(vocab_size, out_size, gpu_id=args.cuda)
    bilstmcrf_model.train(train_word_lists, train_tag_lists,
                          dev_word_lists, dev_tag_lists, word2id, tag2id, debug=args.debug)
    torch.save(bilstmcrf_model, '../data/trained_model/{}'.format(model_name))
    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format('bilstm-crf'))
    pred_tag_lists, test_tag_lists = bilstmcrf_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    print(len(test_tag_lists))
    print(len(pred_tag_lists))
    if args.debug=='1':
        metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=False)
        metrics.report_scores()
        metrics.report_confusion_matrix()


def train_kfold(excel_path, model_name, kfold, tag_version='bio'):

    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e == '<end>': continue
                if e not in maps:
                    maps[e] = len(maps)
        maps['<unk>'] = len(maps)
        maps['<pad>'] = len(maps)
        maps['<start>'] = len(maps)
        maps['<end>'] = len(maps)
        return maps

    # data
    # print("读取数据...")
    with open(r'../data/file/train_{}.json'.format(tag_version)) as f:
        train_dict = json.load(f)
    raw_data = pd.read_excel(excel_path)
    question_list = raw_data['用户问题'].tolist()
    question_list = [question.replace(' ', '') for question in question_list]
    for i in range(len(question_list)):
        question = question_list[i]
        question = question.replace('三十', '30')
        question = question.replace('四十', '40')
        question = question.replace('十块', '10块')
        question = question.replace('六块', '6块')
        question = question.replace('一个月', '1个月')
        question = question.replace('2O', '20')
        question_list[i] = question
    # 按照划分好的id划分数据
    # train_list = [question_list[i] for i in id_dict['train_ids']]
    # dev_list = [question_list[i] for i in id_dict['dev_ids']]
    # test_list = [question_list[i] for i in id_dict['dev_ids']]
    # 随机划分数据
    random.shuffle(question_list)
    block_len = len(question_list)//kfold
    data_blocks = [question_list[i*block_len: (i+1)*block_len] for i in range(kfold-1)]
    data_blocks.append(question_list[(kfold-1)*block_len:])
    # 使用所有训练数据生成word2id tag2id
    all_train_list = []
    for block in data_blocks:
        all_train_list += block
    all_train_word_lists, all_train_tag_lists = process_data(all_train_list, train_dict)
    all_word2id = build_map(all_train_word_lists)
    all_tag2id = build_map(all_train_tag_lists)
    with open(r'../data/file/word2id_{}.json'.format('all'), 'w') as f: json.dump(all_word2id, f)
    with open(r'../data/file/tag2id_{}_{}.json'.format(tag_version, 'all'), 'w') as f: json.dump(all_tag2id, f)
    # 开始kfold训练
    for i, block in enumerate(data_blocks):
        train_list, dev_list = [], block
        for _ in range(kfold):
            if _ != i:
                train_list += data_blocks[_]
        train_word_lists, train_tag_lists = process_data(train_list, train_dict)
        dev_word_lists, dev_tag_lists = process_data(dev_list, train_dict)
        # 生成word2id 和 tag2id 并保存
        word2id = build_map(train_word_lists)
        tag2id = build_map(train_tag_lists)
        with open(r'../data/file/word2id_{}.json'.format(i), 'w') as f: json.dump(word2id, f)
        with open(r'../data/file/tag2id_{}_{}.json'.format(tag_version, i), 'w') as f: json.dump(tag2id, f)
        # train
        print("正在训练评估Bi-LSTM+CRF模型...")
        start = time.time()
        vocab_size = len(all_word2id)
        out_size = len(all_tag2id)
        bilstmcrf_model = BILSTM_CRF_Model(vocab_size, out_size, gpu_id=args.cuda)
        bilstmcrf_model.train(train_word_lists, train_tag_lists,
                              dev_word_lists, dev_tag_lists, all_word2id, all_tag2id, debug=args.debug)
        torch.save(bilstmcrf_model, '../data/trained_model/{}'.format(model_name.replace('.pth', '_{}'.format(i)+'.pth')))
        print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
        if args.debug == '1':
            print("评估{}模型中...".format('bilstm-crf'))
            pred_tag_lists, dev_tag_lists = bilstmcrf_model.test(
                dev_word_lists, dev_tag_lists, all_word2id, all_tag2id)


            print(len(dev_tag_lists))
            print(len(pred_tag_lists))
            metrics = Metrics(dev_tag_lists, pred_tag_lists, remove_O=False)
            metrics.report_scores()
            metrics.report_confusion_matrix()


def train2(excel_path, model_name, tag_version='bio'):

    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e == '<end>': continue
                if e not in maps:
                    maps[e] = len(maps)
        maps['<unk>'] = len(maps)
        maps['<pad>'] = len(maps)
        maps['<start>'] = len(maps)
        maps['<end>'] = len(maps)
        return maps

    # data
    print("读取数据...")
    with open(r'../data/file/train_{}.json'.format(tag_version)) as f:
            train_dict = json.load(f)
    with open(r'../data/file/train_dev_ids.json') as f:
        id_dict = json.load(f)
    raw_data = pd.read_excel(excel_path)
    question_list = raw_data['用户问题'].tolist()
    question_list = [question.replace(' ', '') for question in question_list]
    for i in range(len(question_list)):
        question = question_list[i]
        question = question.replace('三十', '30')
        question = question.replace('四十', '40')
        question = question.replace('十块', '10块')
        question = question.replace('六块', '6块')
        question = question.replace('一个月', '1个月')
        question = question.replace('2O', '20')
        question_list[i] = question
    # 按照划分好的id划分数据
    # train_list = [question_list[i] for i in id_dict['train_ids']]
    # dev_list = [question_list[i] for i in id_dict['dev_ids']]
    # test_list = [question_list[i] for i in id_dict['dev_ids']]
    # 随机划分数据
    random.shuffle(question_list)
    train_list = question_list[:int(0.8*len(question_list))]
    dev_list = question_list[int(0.8*len(question_list)): int(0.9*len(question_list))]
    test_list = question_list[int(0.9*len(question_list)):]

    train_word_lists, train_tag_lists = process_data(train_list, train_dict)
    dev_word_lists, dev_tag_lists = process_data(dev_list, train_dict)
    test_word_lists, test_tag_lists = process_data(test_list, train_dict, test=True)
    # bert
    BERT_PRETRAINED_PATH = '../data/bert_pretrained_model'
    word2id = {}
    with open(os.path.join(BERT_PRETRAINED_PATH, 'vocab.txt'), 'r') as f:
        count = 0
        for line in f:
            word2id[line.split('\n')[0]] = count
            count += 1
    bert_model = BertModel.from_pretrained(BERT_PRETRAINED_PATH)
    # 生成word2id 和 tag2id 并保存
    tag2id = build_map(train_tag_lists)
    with open(r'../data/file/word2id_bert.json', 'w') as f: json.dump(word2id, f)
    with open(r'../data/file/tag2id_{}.json'.format(tag_version), 'w') as f: json.dump(tag2id, f)

    # train
    print("正在训练评估Bert-Bi-LSTM+CRF模型...")
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstmcrf_model = BERT_BILSTM_CRF_Model(vocab_size, out_size, bert_model, gpu_id=args.cuda)
    bilstmcrf_model.train(train_word_lists, train_tag_lists,
                          dev_word_lists, dev_tag_lists, word2id, tag2id)
    torch.save(bilstmcrf_model, '../data/trained_model/{}'.format(model_name))
    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format('bert-bilstm-crf'))
    pred_tag_lists, test_tag_lists = bilstmcrf_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    print(len(test_tag_lists))
    print(len(pred_tag_lists))
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=False)
    metrics.report_scores()
    metrics.report_confusion_matrix()


def evaluate(excel_path, word2id_version='word2id.json', model_name='bilstmcrf.pth', tag_version='bio'):
    with open(r'../data/file/train_{}.json'.format(tag_version), 'r') as f:
        train_dict = json.load(f)
    with open(r'../data/file/{}'.format(word2id_version), 'r') as f: word2id = json.load(f)
    with open(r'../data/file/tag2id_{}.json'.format(tag_version), 'r') as f: tag2id = json.load(f)
    # 得到训练数据
    raw_data = pd.read_excel(excel_path)
    question_list = raw_data['用户问题'].tolist()
    question_list = [question.replace(' ', '') for question in question_list]
    for i in range(len(question_list)):
        question = question_list[i]
        question = question.replace('三十', '30')
        question = question.replace('四十', '40')
        question = question.replace('十块', '10块')
        question = question.replace('六块', '6块')
        question = question.replace('一个月', '1个月')
        question = question.replace('2O', '20')
        question_list[i] = question
    index_list = list(range(len(question_list)))
    random.shuffle(index_list)
    question_list = [question_list[i] for i in index_list]
    test_word_lists, test_tag_lists = process_data(question_list[:], train_dict)
    # 得到原来的约束属性名和约束值
    con_names = raw_data['约束属性名'].tolist()
    con_values = raw_data['约束属性值'].tolist()
    con_names = [re.split(r'[｜\|]', str(con_names[i])) for i in index_list]
    con_values = [re.split(r'[｜\|]', str(con_values[i])) for i in index_list]
    con_dict_list = []
    for i in range(len(con_names)):
        con_dict = {}
        con_name = con_names[i]
        con_value = con_values[i]
        for j in range(len(con_name)):
            name = con_name[j].strip()
            value = con_value[j].strip()
            if name == 'nan' or name == '有效期': continue
            if name == '价格' and value == '1' and '最' in question_list[i]: continue
            if name == '流量' and value == '1' and '最' in question_list[i]: continue
            if name not in con_dict: con_dict[name] = []
            con_dict[name].append(value)
        con_dict_list.append(con_dict)

    service_names = get_all_service(raw_data)

    bilstmcrf_model = load_model(r'../data/trained_model/{}'.format(model_name), map_location=lambda storage, loc: storage.cuda('cuda:{}'.format(args.cuda)))
    bilstmcrf_model.device = torch.device('cuda:{}'.format(args.cuda))
    pred_tag_lists, test_tag_lists = bilstmcrf_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    c = 0
    for index in range(len(pred_tag_lists)):
        pre_dict = get_anno_dict(question_list[index], pred_tag_lists[index], service_names)
        # 对比 官方标注 和 非增强预测标注
        if set(pre_dict) != set(con_dict_list[index]):
            c+=1
            print(question_list[index])
            print('true:',test_tag_lists[index])
            print('pred:',pred_tag_lists[index])
            print('true:',con_dict_list[index])
            print('pred:',pre_dict)
    print(c)


def evaluate_kfold(excel_path, kfold, word2id_version='word2id.json', model_name='bilstmcrf.pth', tag_version='bio'):
    for i in range(kfold):
        with open(r'../data/file/train_{}.json'.format(tag_version), 'r') as f:
            train_dict = json.load(f)
        with open(r'../data/file/{}'.format(word2id_version.replace('.json', '_{}.json'.format('all'))), 'r') as f: word2id = json.load(f)
        with open(r'../data/file/tag2id_{}_{}.json'.format(tag_version, 'all'), 'r') as f: tag2id = json.load(f)
        # 得到训练数据
        raw_data = pd.read_excel(excel_path)
        question_list = raw_data['用户问题'].tolist()
        question_list = [question.replace(' ', '') for question in question_list]
        for i in range(len(question_list)):
            question = question_list[i]
            question = question.replace('三十', '30')
            question = question.replace('四十', '40')
            question = question.replace('十块', '10块')
            question = question.replace('六块', '6块')
            question = question.replace('一个月', '1个月')
            question = question.replace('2O', '20')
            question_list[i] = question
        index_list = list(range(len(question_list)))
        random.shuffle(index_list)
        question_list = [question_list[i] for i in index_list]
        test_word_lists, test_tag_lists = process_data(question_list[:], train_dict)
        # 得到原来的约束属性名和约束值
        con_names = raw_data['约束属性名'].tolist()
        con_values = raw_data['约束属性值'].tolist()
        con_names = [re.split(r'[｜\|]', str(con_names[i])) for i in index_list]
        con_values = [re.split(r'[｜\|]', str(con_values[i])) for i in index_list]
        con_dict_list = []
        for i in range(len(con_names)):
            con_dict = {}
            con_name = con_names[i]
            con_value = con_values[i]
            for j in range(len(con_name)):
                name = con_name[j].strip()
                value = con_value[j].strip()
                if name == 'nan' or name == '有效期': continue
                if name == '价格' and value == '1' and '最' in question_list[i]: continue
                if name == '流量' and value == '1' and '最' in question_list[i]: continue
                if name not in con_dict: con_dict[name] = []
                con_dict[name].append(value)
            con_dict_list.append(con_dict)

        service_names = get_all_service(raw_data)

        bilstmcrf_model = load_model(r'../data/trained_model/{}'.format(model_name.replace('.pth', '_{}.pth'.format(i))), map_location=lambda storage, loc: storage.cuda('cuda:{}'.format(args.cuda)))
        bilstmcrf_model.device = torch.device('cuda:{}'.format(args.cuda))
        pred_tag_lists, test_tag_lists = bilstmcrf_model.test(
            test_word_lists, test_tag_lists, word2id, tag2id)
        # 使用了 enhence 又需要对比增强前的
        c = 0
        for index in range(len(pred_tag_lists)):
            pre_dict = get_anno_dict(question_list[index], pred_tag_lists[index], service_names)
            test_dict = get_anno_dict(question_list[index], test_tag_lists[index][:-1], service_names)
            # 对比 官方标注 和 非增强预测标注
            if set(pre_dict) != set(con_dict_list[index]):
                c+=1
                print(question_list[index])
                print('true:',test_tag_lists[index])
                print('pred:',pred_tag_lists[index])
                print('true:',con_dict_list[index])
                print('pred:',pre_dict)
        print(c)


if __name__ == '__main__':
    print('开始train_ner')
    setup_seed(1)
    excel_path = r'../data/raw_data/train_denoised_ner.xlsx'
    # excel_path = r'../data/raw_data/train_syn.xlsx'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # create_test_BIEO(excel_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', '-tt', default='lstm', type=str)
    parser.add_argument('--tag_version', '-tv', default='bieo', type=str)
    parser.add_argument('--model_name', '-m', default='ner_model_test.pth', type=str)
    parser.add_argument('--cuda', '-c', default=0)
    parser.add_argument('--debug', '-d', default='1', choices=['0', '1'])
    parser.add_argument('--kfold', '-k', default=5, type=int)
    # parser.add_argument('--kfold', '-k', action='store_true')
    args = parser.parse_args()
    a = args.kfold
    if args.tag_version == 'bio':
        create_test_BIO(excel_path)
    else:
        create_test_BIEO(excel_path)
        # statistic_wrong_cons_bieo()
    if args.train_type == 'lstm':
        if args.kfold == 1:
            train(excel_path, model_name=args.model_name, tag_version=args.tag_version)
            if args.debug == '1':
                evaluate(excel_path, word2id_version='word2id.json', model_name=args.model_name, tag_version=args.tag_version)
        else:
            train_kfold(excel_path, model_name=args.model_name, kfold=args.kfold, tag_version=args.tag_version)
            if args.debug == '1':
                evaluate_kfold(excel_path, kfold=args.kfold, word2id_version='word2id.json', model_name=args.model_name,
                         tag_version=args.tag_version)
    else:
        train2(excel_path, model_name=args.model_name, tag_version=args.tag_version)
        if args.debug == '1':
            evaluate(excel_path, word2id_version='word2id_bert.json', model_name=args.model_name, tag_version=args.tag_version)