"""
@Author: hezf 
@Time: 2021/6/9 13:29 
@desc: 推理模块，用于测试集的预测
"""
import warnings
warnings.filterwarnings('ignore')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
import torch
from torch.utils.data import DataLoader
from data import BertDataset, pred_collate_fn, PredDataset, process_data, get_anno_dict, LabelHub, PredEnsembleDataset, \
    process_syn, process_postdo, process_postdo_last
from transformers import BertTokenizer, AutoTokenizer
import argparse
from tqdm.auto import tqdm
from triples import KnowledgeGraph
import json
import pandas as pd
from statistic import get_all_service
import collections

BERT_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/bert-base-chinese'
XLNET_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-xlnet-base'
ROBERTA_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-roberta-wwm-ext'
ELECTRA_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-electra-180g-base-discriminator'
ALBERT_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/albert-chinese-tiny'
GPT2_SMALL_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/gpt2-chinese-small'
GPT2_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/gpt2-chinese-base'
MACBERT_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-macbert-base'


def predict():
    """
    1 加载分类、NER模型
    2 同一例数据用不同模型来预测
    3 获取各个模型的结果，拼接成SPARQL查询语句，并查询得到最终结果
    :return:
    """
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    cls_model = load_model(os.path.join(args.model_path, args.cls_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device)
    ner_model = load_model(os.path.join(args.model_path, args.ner_model), map_location=lambda storage, loc: storage.cuda(args.cuda))
    ner_model.device = device
    # data
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
    pred_dataset = PredDataset(args.data_path, tokenizer)
    dataloader = DataLoader(pred_dataset, batch_size=args.batch_size, collate_fn=pred_collate_fn)
    with open(r'../data/file/word2id.json', 'r') as f: word2id = json.load(f)
    with open(r'../data/file/tag2id_{}.json'.format(args.tag_version), 'r') as f: tag2id = json.load(f)
    raw_data = pd.read_excel(r'../data/raw_data/train_denoised.xlsx')
    service_names = get_all_service(raw_data)
    label_hub = LabelHub(label_file='../data/dataset/cls_label2id.json')
    kg = KnowledgeGraph(rdf_file='../data/process_data/triples.rdf')
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.cuda)
    # answer
    ans = {'result': {}, 'model_result': {}}
    empty_ans = []
    view_gpu_info(args.cuda)
    for input_ids, token_type_ids, attention_mask, questions in tqdm_with_debug(dataloader, debug=args.debug):
        input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
        ans_logits, prop_logits, entity_logits = cls_model(input_ids=input_ids,
                                                       token_type_ids=token_type_ids,
                                                       attention_mask=attention_mask)
        test_word_lists = process_data(questions)
        test_tag_lists = [[] for _ in range(len(test_word_lists))]
        ner_logits, _ = ner_model.test(test_word_lists, test_tag_lists, word2id, tag2id)
        # 处理中间结果
        ans_pred = ans_logits.data.cpu().argmax(dim=-1)
        prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
        entity_pred = entity_logits.data.cpu().argmax(dim=-1)
        ans_labels, prop_labels, entity_labels = get_labels(ans_pred, prop_pred, entity_pred, label_hub)
        # 获得答案
        for i, question in enumerate(questions):
            # 处理ner结果，得到 main_property 和 sub_properties
            question = question.replace(' ', '_')
            pre_dict = get_anno_dict(questions[i], ner_logits[i], service_names)
            sub_properties = []
            for k, v in pre_dict.items():
                for vi in range(len(v)):
                    sub_properties.append((k, v[vi]))
            # 过滤 sub_properties
            sub_properties_filter = filter_sub_properties(sub_properties, entity_labels[i], ans_labels[i])
            sub_properties_filter = process_postdo(sub_properties_filter, question, entity_labels[i])
            # 得到 operator
            opr, obj = get_operator(question)
            if opr is None:
                operator = 'other'
            else:
                operator = opr
                sub_properties_filter = [(obj, 1)]
            # 使用规则来判断"属性名"，判断成功则直接覆盖模型预测结果
            prop_label = rules_to_judge_prop(question)
            if prop_label is not None:
                prop_labels[i] = [prop_label]
            entity_labels[i], prop_labels[i] = process_postdo_last(question, entity_labels[i], prop_labels[i])
            # 模型预测的结果
            model_result = {
                'question': question,
                'ans_type': ans_labels[i],
                'entity': entity_labels[i],
                'main_property': prop_labels[i],
                'operator': operator,
                'sub_properties': sub_properties_filter}
            temp_ans = kg.fetch_ans(**model_result)
            if len(temp_ans) == 0:
                # 1 答案为空且出现意思相近的实体时
                changed, entity_label = reverse_entity(entity_labels[i])
                if changed:
                    ori_entity = model_result['entity']
                    model_result['entity'] = entity_label
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['entity'] = ori_entity
                # 2 答案为空且出现意思相近的属性名时
                if len(temp_ans) == 0 and reverse_prop(prop_labels[i]):
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = prop_labels[i]
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
                # 3 答案为空且约束属性名不为空时，尽最大可能找答案
                if len(temp_ans) == 0 and len(sub_properties_filter) > 0:
                    model_result['sub_properties'] = []
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['sub_properties'] = sub_properties_filter
                # 4 属性名换成“档位介绍-业务简介”
                if len(temp_ans) == 0:
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = ['档位介绍-业务简介']
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
            if len(temp_ans) == 0:
                empty_ans.append(model_result)
            for j in range(len(temp_ans)):
                temp_ans[j] = str(temp_ans[j])
            ans['model_result'][str(len(ans['model_result']))] = model_result
            ans['result'][str(len(ans['result']))] = '|'.join(temp_ans)
    # 读取真实结果，计算得分
    if args.dev == '1':
        with open('../data/dataset/cls_dev.txt', 'r', encoding='utf-8') as f:
            answer = {}
            for i, line in enumerate(f):
                one_ans = line.strip().split('\t')[-1]
                answer[str(i)] = one_ans
        # print('\n验证结果是：', score_evalution(answers=answer, predictions=ans['result']))
        with open('../data/results/{}_dev.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)
    # 直接写入答案
    else:
        if args.debug == '1':
            with open('../data/results/{}.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
               json.dump(ans, f, ensure_ascii=False, indent=2)
            generate_false_label('{}.json'.format(args.cls_model.replace(':', '：')))
        else:
            with open('../code/result.json', 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
    print('答案写入成功，空结果有{}个。'.format(empty_ans))


def predict_kfold():
    """
    1 加载分类、NER模型
    2 同一例数据用不同模型来预测
    3 获取各个模型的结果，拼接成SPARQL查询语句，并查询得到最终结果
    :return:
    """
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    cls_model = load_model(os.path.join(args.model_path, args.cls_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device)
    ner_model_list = []
    for i in range(args.kfold):
        ner_model = load_model(os.path.join(args.model_path, args.ner_model.replace('.pth', '_{}.pth'.format(i))), map_location=lambda storage, loc: storage.cuda(args.cuda))
        ner_model.device = device
        ner_model_list.append(ner_model)
    # data
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
    pred_dataset = PredDataset(args.data_path, tokenizer)
    dataloader = DataLoader(pred_dataset, batch_size=args.batch_size, collate_fn=pred_collate_fn)
    word2id_list = []
    tag2id_list = []
    for i in range(args.kfold):
        with open(r'../data/file/word2id_{}.json'.format('all'), 'r') as f:
            word2id = json.load(f)
            word2id_list.append(word2id)
        with open(r'../data/file/tag2id_bieo_{}.json'.format('all'), 'r') as f:
            tag2id = json.load(f)
            tag2id_list.append(tag2id)
    raw_data = pd.read_excel(r'../data/raw_data/train_denoised.xlsx')
    service_names = get_all_service(raw_data)
    label_hub = LabelHub(label_file='../data/dataset/cls_label2id.json')
    kg = KnowledgeGraph(rdf_file='../data/process_data/triples.rdf')
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.cuda)
    # answer
    ans = {'result': {}, 'model_result': {}}
    empty_ans = []
    view_gpu_info(args.cuda)
    for input_ids, token_type_ids, attention_mask, questions in tqdm(dataloader):
        input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
        ans_logits, prop_logits, entity_logits = cls_model(input_ids=input_ids,
                                                       token_type_ids=token_type_ids,
                                                       attention_mask=attention_mask)
        # 处理中间结果
        ans_pred = ans_logits.data.cpu().argmax(dim=-1)
        prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
        entity_pred = entity_logits.data.cpu().argmax(dim=-1)
        ans_labels, prop_labels, entity_labels = get_labels(ans_pred, prop_pred, entity_pred, label_hub)
        # ner
        questions = process_syn(questions, entity_labels, ans_labels)
        test_word_lists = process_data(questions)
        test_tag_lists = [[] for _ in range(len(test_word_lists))]
        # kfold 决策方法
        # ner_logits_list = []
        # for i in range(args.kfold):
        #     ner_logits, _ = ner_model_list[i].test(test_word_lists, test_tag_lists, word2id_list[i], tag2id_list[i])
        #     ner_logits_list.append(ner_logits)
        # ner avr
        ner_score_list = []
        for i in range(args.kfold):
            ner_score = ner_model_list[i].test_k_pre(test_word_lists, test_tag_lists, word2id_list[i])
            ner_score_list.append(ner_score)
        avg_score = torch.mean(torch.stack(ner_score_list), dim=0)
        ner_logits, _ = ner_model_list[0].test_k_post(test_word_lists, test_tag_lists, word2id_list[0], tag2id_list[0],
                                                   avg_score)
        # 获得答案
        for i, question in enumerate(questions):
            # 处理ner结果，得到 main_property 和 sub_properties
            question = question.replace(' ', '_')
            # ner 5-fold 平均值方式
            pre_dict = get_anno_dict(questions[i], ner_logits[i], service_names)
            # ner 5-fold 预测结果 并统计次数
            # temp_dict = collections.defaultdict(lambda: 0)
            # for fold in range(args.kfold):
            #     pre_dict = get_anno_dict(questions[i], ner_logits_list[fold][i], service_names)
            #     for name, values in pre_dict.items():
            #         for value in values:
            #             temp_dict[name + '_' + value] += 1
            # # ner 5-fold 决策 加入次数大于2的，去掉次数小于等于2的
            # for n, v in temp_dict.items():
            #     if v < 5:
            #         name, value = n.split('_')
            #         if v > 2:
            #             if name not in pre_dict:
            #                 pre_dict[name] = [value]
            #             else:
            #                 if value not in pre_dict[name]:
            #                     pre_dict[name].append(value)
            #         else:
            #             if name in pre_dict and value in pre_dict[name]:
            #                 pre_dict[name].remove(value)
            #                 if len(pre_dict[name]) == 0:
            #                     pre_dict.pop(name)
            #         if args.debug == '1':
            #             print(n, v)
            #             print(question)
            #             print(pre_dict)
            sub_properties = []
            for k, v in pre_dict.items():
                for vi in range(len(v)):
                    sub_properties.append((k, v[vi]))
            # 过滤 sub_properties
            sub_properties_filter = filter_sub_properties(sub_properties, entity_labels[i], ans_labels[i])
            # 比较句出现了3个的情况
            if ans_labels[i] == '比较句':
                if len(sub_properties_filter) == 3:
                    sub_properties_filter = sub_properties_filter[1:]
            sub_properties_filter = process_postdo(sub_properties_filter, question, entity_labels[i])

            # 得到 operator
            opr, obj = get_operator(question)
            if opr is None:
                operator = 'other'
            else:
                operator = opr
                sub_properties_filter = [(obj, 1)]
            # 使用规则来判断"属性名"，判断成功则直接覆盖模型预测结果
            prop_label = rules_to_judge_prop(question)
            if prop_label is not None:
                prop_labels[i] = [prop_label]
            # 模型预测的结果
            model_result = {
                'question': question,
                'ans_type': ans_labels[i],
                'entity': entity_labels[i],
                'main_property': prop_labels[i],
                'operator': operator,
                'sub_properties': sub_properties_filter}
            temp_ans = kg.fetch_ans(**model_result)
            if len(temp_ans) == 0:
                # 1 答案为空且出现意思相近的实体时
                changed, entity_label = reverse_entity(entity_labels[i])
                if changed:
                    ori_entity = model_result['entity']
                    model_result['entity'] = entity_label
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['entity'] = ori_entity
                # 2 答案为空且出现意思相近的属性名时
                if len(temp_ans) == 0 and reverse_prop(prop_labels[i]):
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = prop_labels[i]
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
                # 3 答案为空且约束属性名不为空时，尽最大可能找答案
                if len(temp_ans) == 0 and len(sub_properties_filter) > 0:
                    model_result['sub_properties'] = []
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['sub_properties'] = sub_properties_filter
                # 4 属性名换成“档位介绍-业务简介”
                if len(temp_ans) == 0:
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = ['档位介绍-业务简介']
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
            if len(temp_ans) == 0:
                empty_ans.append(model_result)
            for j in range(len(temp_ans)):
                temp_ans[j] = str(temp_ans[j])
            ans['model_result'][str(len(ans['model_result']))] = model_result
            ans['result'][str(len(ans['result']))] = '|'.join(temp_ans)
    # 读取真实结果，计算得分
    if args.dev == '1':
        with open('../data/dataset/cls_dev.txt', 'r', encoding='utf-8') as f:
            answer = {}
            for i, line in enumerate(f):
                one_ans = line.strip().split('\t')[-2]
                answer[str(i)] = one_ans
        # print('\n验证结果是：', score_evalution(answers=answer, predictions=ans['result']))
        with open('../data/results/{}_dev.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)
    # 直接写入答案
    else:
        if args.debug == '1':
            with open('../data/results/{}.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
               json.dump(ans, f, ensure_ascii=False, indent=2)
            generate_false_label('{}.json'.format(args.cls_model.replace(':', '：')))
        else:
            with open('../code/result.json', 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
    print('答案写入成功，空结果有{}个。'.format(len(empty_ans)))
    for e_ans in empty_ans:
        print(e_ans)

    # 查看预测的效果
    print()
    for id, detail in ans['model_result'].items():
        print(id,end='=')
        print(detail['question'], end='=')
        print(detail['ans_type'], end='=')
        print(detail['entity'], end='=')
        for i in range(len(detail['main_property'])):
            if i < len(detail['main_property']) - 1:
                print(detail['main_property'][i], end='x')
            else:
                print(detail['main_property'][i], end='=')
        if len(detail['main_property']) == 0:
            print(end='=')
        print(detail['operator'], end='=')
        for i in range(len(detail['sub_properties'])):
            if i < len(detail['sub_properties']) - 1:
                print(str(detail['sub_properties'][i][0])+'_'+str(detail['sub_properties'][i][1]), end='x')
            else:
                print(str(detail['sub_properties'][i][0])+'_'+str(detail['sub_properties'][i][1]), end='+')
        if len(detail['sub_properties']) == 0:
            print(end='+')
    print()


def predict_ensemble_one_fold():
    """
    集成模型推理
    :return:
    """
    def tokenize(questions_, tokenizer, device_):
        input_ids_, token_type_ids_, attention_mask_ = [], [], []
        for q in questions_:
            token_dict = tokenizer(q, add_special_tokens=True, max_length=50,
                                   padding='max_length', return_tensors='pt',
                                   truncation=True)
            input_ids_.append(token_dict['input_ids'][0].squeeze(0))
            token_type_ids_.append(token_dict['token_type_ids'][0].squeeze(0))
            attention_mask_.append(token_dict['attention_mask'][0].squeeze(0))
        return {'input_ids': torch.stack(input_ids_).to(device_),
                'token_type_ids': torch.stack(token_type_ids_).to(device_),
                'attention_mask': torch.stack(attention_mask_).to(device_)}

    device = torch.device('cuda:{}'.format(args.cuda))
    # 分类模型载入model: bert、xlnet、electra、
    model_list, tokenizer_list = [], []
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
    model_list.append(load_model(os.path.join(args.model_path, args.cls_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
    tokenizer_list.append(bert_tokenizer)
    if args.xlnet_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.xlnet_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(XLNET_PRETRAINED_PATH))
    if args.electra_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.electra_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(ELECTRA_PRETRAINED_PATH))
    if args.roberta_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.roberta_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(ROBERTA_PRETRAINED_PATH))
    if args.albert_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.albert_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        # albert使用BertTokenizer
        tokenizer_list.append(bert_tokenizer)
    if args.gpt2_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.gpt2_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(GPT2_PRETRAINED_PATH))
    if args.macbert_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.macbert_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(MACBERT_PRETRAINED_PATH))

    # 5折交叉验证的剩余模型
    if args.fold:
        for i in range(1, 5):
            model_name = 'bert_model_{}.pth'.format(i)
            model_list.append(load_model(os.path.join(args.model_path, model_name), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
            tokenizer_list.append(bert_tokenizer)

    # # NER模型载入
    ner_model = load_model(os.path.join(args.model_path, args.ner_model),
                           map_location=lambda storage, loc: storage.cuda(args.cuda))
    ner_model.device = device
    # data
    pred_dataset = PredEnsembleDataset(args.data_path)
    dataloader = DataLoader(pred_dataset, batch_size=args.batch_size)
    # raw_data
    with open(r'../data/file/word2id.json', 'r') as f:
        word2id = json.load(f)
    with open(r'../data/file/tag2id_bieo.json', 'r') as f:
        tag2id = json.load(f)
    raw_data = pd.read_excel(r'../data/raw_data/train_denoised.xlsx')
    service_names = get_all_service(raw_data)
    # support file
    label_hub = LabelHub(label_file='../data/dataset/cls_label2id.json')
    kg = KnowledgeGraph(rdf_file='../data/process_data/triples.rdf')
    # answer
    ans = {'result': {}, 'model_result': {}}
    empty_ans = []
    for idx, questions in tqdm_with_debug(enumerate(dataloader), debug=args.debug):
        ans_logits, prop_logits, entity_logits = [], [], []
        for i, model in enumerate(model_list):
            logits = model(**tokenize(questions, tokenizer_list[i], device))
            ans_logits.append(logits[0])
            prop_logits.append(logits[1])
            entity_logits.append(logits[2])
        # 以下两种融合方式二选一
        # 分类模型平均融合
        ans_labels, prop_labels, entity_labels = average_integration(ans_logits, prop_logits, entity_logits, label_hub)
        # # 分类模型投票融合:效果不好，已被弃用
        # ans_labels, prop_labels, entity_labels = vote_integration(ans_logits, prop_logits, entity_logits, label_hub, args.batch_size)
        # ner标签获取
        # questions = process_syn(questions, entity_labels, ans_labels)
        test_word_lists = process_data(questions)
        test_tag_lists = [[] for _ in range(len(test_word_lists))]
        ner_logits, _ = ner_model.test(test_word_lists, test_tag_lists, word2id, tag2id)
        # 获得答案
        for i, question in enumerate(questions):
            # 处理ner结果，得到 main_property 和 sub_properties
            question = question.replace(' ', '_')
            pre_dict = get_anno_dict(questions[i], ner_logits[i], service_names)
            sub_properties = []
            for k, v in pre_dict.items():
                for vi in range(len(v)):
                    sub_properties.append((k, v[vi]))
            # 过滤 sub_properties
            sub_properties_filter = filter_sub_properties(sub_properties, entity_labels[i], ans_labels[i])
            if ans_labels[i] == '比较句':
                if len(sub_properties_filter) == 3:
                    sub_properties_filter = sub_properties_filter[:2]
            # sub_properties_filter = process_postdo(sub_properties_filter, question, entity_labels[i])
            # 得到 operator
            opr, obj = get_operator(question)
            if opr is None:
                operator = 'other'
            else:
                operator = opr
                sub_properties_filter = [(obj, 1)]
            # 使用规则来判断"属性名"，判断成功则直接覆盖模型预测结果
            prop_label = rules_to_judge_prop(question)
            if prop_label is not None:
                prop_labels[i] = [prop_label]
            # 模型预测的结果
            model_result = {
                'question': question,
                'ans_type': ans_labels[i],
                'entity': entity_labels[i],
                'main_property': prop_labels[i],
                'operator': operator,
                'sub_properties': sub_properties_filter}
            temp_ans = kg.fetch_ans(**model_result)
            if len(temp_ans) == 0:
                # 1 答案为空且出现意思相近的实体时
                changed, entity_label = reverse_entity(entity_labels[i])
                if changed:
                    ori_entity = model_result['entity']
                    model_result['entity'] = entity_label
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['entity'] = ori_entity
                # 2 答案为空且出现意思相近的属性名时
                if len(temp_ans) == 0 and reverse_prop(prop_labels[i]):
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = prop_labels[i]
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
                # 3 答案为空且约束属性名不为空时，尽最大可能找答案
                if len(temp_ans) == 0 and len(sub_properties_filter) > 0:
                    model_result['sub_properties'] = []
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['sub_properties'] = sub_properties_filter
                # 4 属性名换成“档位介绍-业务简介”
                if len(temp_ans) == 0:
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = ['档位介绍-业务简介']
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
            if len(temp_ans) == 0:
                empty_ans.append(model_result)
            for j in range(len(temp_ans)):
                temp_ans[j] = str(temp_ans[j])
            ans['model_result'][str(len(ans['model_result']))] = model_result
            ans['result'][str(len(ans['result']))] = '|'.join(temp_ans)
    # 读取真实结果，计算得分
    if args.dev == '1':
        with open('../data/dataset/cls_dev.txt', 'r', encoding='utf-8') as f:
            answer = {}
            for i, line in enumerate(f):
                one_ans = line.strip().split('\t')[-2]
                answer[str(i)] = one_ans
        # print('\n验证结果是：', score_evalution(answers=answer, predictions=ans['result']))
        with open('../data/results/ensemble_{}_dev.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)
    # 直接写入答案
    else:
        if args.debug == '1':
            with open('../data/results/ensemble_{}.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
            generate_false_label('ensemble_{}.json'.format(args.cls_model.replace(':', '：')))
        else:
            with open('/code/result.json', 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
    print('答案写入成功，空结果有{}个。'.format(len(empty_ans)))
    for e_ans in empty_ans:
        print(e_ans)

    # 查看预测的效果
    print()
    for id, detail in ans['model_result'].items():
        print(id, end='=')
        print(detail['question'], end='=')
        print(detail['ans_type'], end='=')
        print(detail['entity'], end='=')
        for i in range(len(detail['main_property'])):
            if i < len(detail['main_property']) - 1:
                print(detail['main_property'][i], end='x')
            else:
                print(detail['main_property'][i], end='=')
        if len(detail['main_property']) == 0:
            print(end='=')
        print(detail['operator'], end='=')
        for i in range(len(detail['sub_properties'])):
            if i < len(detail['sub_properties']) - 1:
                print(str(detail['sub_properties'][i][0]) + '_' + str(detail['sub_properties'][i][1]), end='x')
            else:
                print(str(detail['sub_properties'][i][0]) + '_' + str(detail['sub_properties'][i][1]), end='+')
        if len(detail['sub_properties']) == 0:
            print(end='+')
    print()


def predict_ensemble():
    """
    集成模型推理
    :return:
    """
    def tokenize(questions_, tokenizer, device_):
        input_ids_, token_type_ids_, attention_mask_ = [], [], []
        for q in questions_:
            token_dict = tokenizer(q, add_special_tokens=True, max_length=50,
                                   padding='max_length', return_tensors='pt',
                                   truncation=True)
            input_ids_.append(token_dict['input_ids'][0].squeeze(0))
            token_type_ids_.append(token_dict['token_type_ids'][0].squeeze(0))
            attention_mask_.append(token_dict['attention_mask'][0].squeeze(0))
        return {'input_ids': torch.stack(input_ids_).to(device_),
                'token_type_ids': torch.stack(token_type_ids_).to(device_),
                'attention_mask': torch.stack(attention_mask_).to(device_)}

    device = torch.device('cuda:{}'.format(args.cuda))
    # 分类模型载入model: bert、xlnet、electra、
    model_list, tokenizer_list = [], []
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
    model_list.append(load_model(os.path.join(args.model_path, args.cls_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
    tokenizer_list.append(bert_tokenizer)
    if args.xlnet_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.xlnet_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(XLNET_PRETRAINED_PATH))
    if args.electra_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.electra_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(ELECTRA_PRETRAINED_PATH))
    if args.roberta_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.roberta_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(ROBERTA_PRETRAINED_PATH))
    if args.albert_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.albert_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        # albert使用BertTokenizer
        tokenizer_list.append(bert_tokenizer)
    if args.gpt2_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.gpt2_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(GPT2_PRETRAINED_PATH))
    if args.macbert_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.macbert_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(MACBERT_PRETRAINED_PATH))

    # 5折交叉验证的剩余模型
    if args.fold:
        for i in range(1, 5):
            model_name = 'bert_model_{}.pth'.format(i)
            model_list.append(load_model(os.path.join(args.model_path, model_name), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
            tokenizer_list.append(bert_tokenizer)

    # # NER模型载入
    ner_model_list = []
    for i in range(args.kfold):
        ner_model = load_model(os.path.join(args.model_path, args.ner_model.replace('.pth', '_{}.pth'.format(i))),
                               map_location=lambda storage, loc: storage.cuda(args.cuda))
        ner_model.device = device
        ner_model_list.append(ner_model)
    # data
    pred_dataset = PredEnsembleDataset(args.data_path)
    dataloader = DataLoader(pred_dataset, batch_size=args.batch_size)
    # raw_data
    word2id_list = []
    tag2id_list = []
    for i in range(args.kfold):
        with open(r'../data/file/word2id_{}.json'.format('all'), 'r') as f:
            word2id = json.load(f)
            word2id_list.append(word2id)
        with open(r'../data/file/tag2id_bieo_{}.json'.format('all'), 'r') as f:
            tag2id = json.load(f)
            tag2id_list.append(tag2id)
    raw_data = pd.read_excel(r'../data/raw_data/train_denoised.xlsx')
    service_names = get_all_service(raw_data)
    # support file
    label_hub = LabelHub(label_file='../data/dataset/cls_label2id.json')
    kg = KnowledgeGraph(rdf_file='../data/process_data/triples.rdf')
    # answer
    ans = {'result': {}, 'model_result': {}}
    empty_ans = []
    for idx, questions in tqdm_with_debug(enumerate(dataloader), debug=args.debug):
        ans_logits, prop_logits, entity_logits = [], [], []
        for i, model in enumerate(model_list):
            logits = model(**tokenize(questions, tokenizer_list[i], device))
            ans_logits.append(logits[0])
            prop_logits.append(logits[1])
            entity_logits.append(logits[2])
        # 以下两种融合方式二选一
        # 分类模型平均融合
        ans_labels, prop_labels, entity_labels = average_integration(ans_logits, prop_logits, entity_logits, label_hub)
        # # 分类模型投票融合:效果不好，已被弃用
        # ans_labels, prop_labels, entity_labels = vote_integration(ans_logits, prop_logits, entity_logits, label_hub, args.batch_size)
        # ner标签获取
        questions = process_syn(questions, entity_labels, ans_labels)
        test_word_lists = process_data(questions)
        test_tag_lists = [[] for _ in range(len(test_word_lists))]
        # kfold 决策方法
        # ner_logits_list = []
        # for i in range(args.kfold):
        #     ner_logits, _ = ner_model_list[i].test(test_word_lists, test_tag_lists, word2id_list[i], tag2id_list[i])
        #     ner_logits_list.append(ner_logits)
        # kfold 平均分方法
        ner_score_list = []
        for i in range(args.kfold):
            ner_score = ner_model_list[i].test_k_pre(test_word_lists, test_tag_lists, word2id_list[i])
            ner_score_list.append(ner_score)
        avg_score = torch.mean(torch.stack(ner_score_list), dim=0)
        ner_logits, _ = ner_model_list[0].test_k_post(test_word_lists, test_tag_lists, word2id_list[0], tag2id_list[0], avg_score)
        # 获得答案
        for i, question in enumerate(questions):
            # 处理ner结果，得到 main_property 和 sub_properties
            question = question.replace(' ', '_')
            # ner 5-fold 平均值方式
            pre_dict = get_anno_dict(questions[i], ner_logits[i], service_names)
            # ner 5-fold 预测结果 并统计次数
            # temp_dict = collections.defaultdict(lambda: 0)
            # for fold in range(args.kfold):
            #     pre_dict = get_anno_dict(questions[i], ner_logits_list[fold][i], service_names)
            #     for name, values in pre_dict.items():
            #         for value in values:
            #             temp_dict[name + '_' + value] += 1
            # # ner 5-fold 决策 加入次数大于2的，去掉次数小于等于2的
            # for n, v in temp_dict.items():
            #     if v < 5:
            #         name, value = n.split('_')
            #         if v > 2:
            #             if name not in pre_dict:
            #                 pre_dict[name] = [value]
            #             else:
            #                 if value not in pre_dict[name]:
            #                     pre_dict[name].append(value)
            #         else:
            #             if name in pre_dict and value in pre_dict[name]:
            #                 pre_dict[name].remove(value)
            #                 if len(pre_dict[name]) == 0:
            #                     pre_dict.pop(name)
            #         print(n, v)
            #         print(question)
            #         print(pre_dict)
            # 整理ner的预测结果格式
            sub_properties = []
            for k, v in pre_dict.items():
                for vi in range(len(v)):
                    sub_properties.append((k, v[vi]))
            # 过滤 sub_properties
            sub_properties_filter = filter_sub_properties(sub_properties, entity_labels[i], ans_labels[i])
            # todo 使用规则方法进行后处理
            if ans_labels[i] == '比较句':
                if len(sub_properties_filter) == 3:
                    sub_properties_filter = sub_properties_filter[1:]
            sub_properties_filter = process_postdo(sub_properties_filter, question, entity_labels[i])
            # 得到 operator
            opr, obj = get_operator(question)
            if opr is None:
                operator = 'other'
            else:
                operator = opr
                sub_properties_filter = [(obj, 1)]
            # 使用规则来判断"属性名"，判断成功则直接覆盖模型预测结果
            prop_label = rules_to_judge_prop(question)
            if prop_label is not None:
                prop_labels[i] = [prop_label]
            # 对实体进行后处理，由于部分实体和同义词容易混淆 在验证集上结果经常会出错，用规则的方式来判断
            entity_labels[i], prop_label[i] = process_postdo_last(question, entity_labels[i], prop_label[i])
            # 模型预测的结果
            model_result = {
                'question': question,
                'ans_type': ans_labels[i],
                'entity': entity_labels[i],
                'main_property': prop_labels[i],
                'operator': operator,
                'sub_properties': sub_properties_filter}
            temp_ans = kg.fetch_ans(**model_result)
            if len(temp_ans) == 0:
                # 1 答案为空且出现意思相近的实体时
                changed, entity_label = reverse_entity(entity_labels[i])
                if changed:
                    ori_entity = model_result['entity']
                    model_result['entity'] = entity_label
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['entity'] = ori_entity
                # 2 答案为空且出现意思相近的属性名时
                if (len(temp_ans) == 0 or (len(temp_ans)==1 and ans_labels[i] == '并列句')) and reverse_prop(prop_labels[i]):
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = prop_labels[i]
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
                # 3 答案为空且约束属性名不为空时，尽最大可能找答案
                if len(temp_ans) == 0 and len(sub_properties_filter) > 0:
                    model_result['sub_properties'] = []
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['sub_properties'] = sub_properties_filter
                # 4 属性名换成“档位介绍-业务简介”
                if len(temp_ans) == 0:
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = ['档位介绍-业务简介']
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
            if len(temp_ans) == 0:
                empty_ans.append(model_result)
            for j in range(len(temp_ans)):
                temp_ans[j] = str(temp_ans[j])
            ans['model_result'][str(len(ans['model_result']))] = model_result
            ans['result'][str(len(ans['result']))] = '|'.join(temp_ans)
    # 读取真实结果，计算得分
    if args.dev == '1':
        with open('../data/dataset/cls_dev.txt', 'r', encoding='utf-8') as f:
            answer = {}
            for i, line in enumerate(f):
                one_ans = line.strip().split('\t')[-2]
                answer[str(i)] = one_ans
        # print('\n验证结果是：', score_evalution(answers=answer, predictions=ans['result']))
        with open('../data/results/ensemble_{}_dev.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)
    # 直接写入答案
    else:
        if args.debug == '1':
            with open('../data/results/ensemble_{}.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
            generate_false_label('ensemble_{}.json'.format(args.cls_model.replace(':', '：')))
        else:
            with open('/code/result.json', 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
    print('答案写入成功，空结果有{}个。'.format(len(empty_ans)))
    for e_ans in empty_ans:
        print(e_ans)

    # 查看预测的效果
    print()
    for id, detail in ans['model_result'].items():
        print(id, end='=')
        print(detail['question'], end='=')
        print(detail['ans_type'], end='=')
        print(detail['entity'], end='=')
        for i in range(len(detail['main_property'])):
            if i < len(detail['main_property']) - 1:
                print(detail['main_property'][i], end='x')
            else:
                print(detail['main_property'][i], end='=')
        if len(detail['main_property']) == 0:
            print(end='=')
        print(detail['operator'], end='=')
        for i in range(len(detail['sub_properties'])):
            if i < len(detail['sub_properties']) - 1:
                print(str(detail['sub_properties'][i][0]) + '_' + str(detail['sub_properties'][i][1]), end='x')
            else:
                print(str(detail['sub_properties'][i][0]) + '_' + str(detail['sub_properties'][i][1]), end='+')
        if len(detail['sub_properties']) == 0:
            print(end='+')
    print()


def predict_ensemble_with_binary():
    """
    集成模型推理
    :return:
    """
    def tokenize(questions_, tokenizer, device_):
        input_ids_, token_type_ids_, attention_mask_ = [], [], []
        for q in questions_:
            token_dict = tokenizer(q, add_special_tokens=True, max_length=50,
                                   padding='max_length', return_tensors='pt',
                                   truncation=True)
            input_ids_.append(token_dict['input_ids'][0].squeeze(0))
            token_type_ids_.append(token_dict['token_type_ids'][0].squeeze(0))
            attention_mask_.append(token_dict['attention_mask'][0].squeeze(0))
        return {'input_ids': torch.stack(input_ids_).to(device_),
                'token_type_ids': torch.stack(token_type_ids_).to(device_),
                'attention_mask': torch.stack(attention_mask_).to(device_)}

    device = torch.device('cuda:{}'.format(args.cuda))
    # 分类模型载入model: bert、xlnet、electra、
    model_list, tokenizer_list = [], []
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
    model_list.append(load_model(os.path.join(args.model_path, args.cls_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
    tokenizer_list.append(bert_tokenizer)
    if args.xlnet_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.xlnet_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(XLNET_PRETRAINED_PATH))
    if args.electra_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.electra_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(ELECTRA_PRETRAINED_PATH))
    if args.roberta_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.roberta_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(ROBERTA_PRETRAINED_PATH))
    if args.albert_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.albert_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        # albert使用BertTokenizer
        tokenizer_list.append(bert_tokenizer)
    if args.gpt2_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.gpt2_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(GPT2_PRETRAINED_PATH))
    if args.macbert_model != '':
        model_list.append(load_model(os.path.join(args.model_path, args.macbert_model),
                                     map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
        tokenizer_list.append(AutoTokenizer.from_pretrained(MACBERT_PRETRAINED_PATH))

    # 5折交叉验证的剩余模型
    if args.fold:
        for i in range(1, 5):
            model_name = 'bert_model_{}.pth'.format(i)
            model_list.append(load_model(os.path.join(args.model_path, model_name), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device))
            tokenizer_list.append(bert_tokenizer)

    # # NER模型载入
    ner_model_list = []
    for i in range(args.kfold):
        ner_model = load_model(os.path.join(args.model_path, args.ner_model.replace('.pth', '_{}.pth'.format(i))),
                               map_location=lambda storage, loc: storage.cuda(args.cuda))
        ner_model.device = device
        ner_model_list.append(ner_model)
    # 二分类模型载入
    binary_model_list, binary_tokenizer_list = [], []
    # for idx in range(5):
    #     binary_model_name = 'binary_bert_{}.pth'.format(idx)
    #     binary_model_list.append(
    #         load_model(os.path.join(args.model_path, binary_model_name), map_location=lambda storage, loc: storage.cuda(args.cuda)))
    #     binary_tokenizer_list.append(bert_tokenizer)
    count = 0
    for file in os.listdir('../data/trained_model/'):
        if 'binary.pth' in file and count < 1:
            count += 1
            binary_model_list.append(load_model(os.path.join(args.model_path, file), map_location=lambda storage, loc: storage.cuda(args.cuda)))
            if 'roberta_binary.pth' == file:
                binary_tokenizer_list.append(AutoTokenizer.from_pretrained(ROBERTA_PRETRAINED_PATH))
            elif 'bert_binary.pth' == file or 'albert_binary.pth' == file:
                binary_tokenizer_list.append(bert_tokenizer)
            elif 'xlnet_binary.pth' == file:
                binary_tokenizer_list.append(AutoTokenizer.from_pretrained(XLNET_PRETRAINED_PATH))
            elif 'macbert_binary.pth' == file:
                binary_tokenizer_list.append(AutoTokenizer.from_pretrained(MACBERT_PRETRAINED_PATH))
    # data
    pred_dataset = PredEnsembleDataset(args.data_path)
    dataloader = DataLoader(pred_dataset, batch_size=args.batch_size)
    # raw_data
    word2id_list = []
    tag2id_list = []
    for i in range(args.kfold):
        with open(r'../data/file/word2id_{}.json'.format('all'), 'r') as f:
            word2id = json.load(f)
            word2id_list.append(word2id)
        with open(r'../data/file/tag2id_bieo_{}.json'.format('all'), 'r') as f:
            tag2id = json.load(f)
            tag2id_list.append(tag2id)
    raw_data = pd.read_excel(r'../data/raw_data/train_denoised.xlsx')
    service_names = get_all_service(raw_data)
    # support file
    label_hub = LabelHub(label_file='../data/dataset/cls_label2id.json')
    kg = KnowledgeGraph(rdf_file='../data/process_data/triples.rdf')
    # answer
    ans = {'result': {}, 'model_result': {}}
    empty_ans = []
    for idx, questions in tqdm_with_debug(enumerate(dataloader), debug=args.debug):
        ans_logits, prop_logits, entity_logits = [], [], []
        for i, model in enumerate(model_list):
            logits = model(**tokenize(questions, tokenizer_list[i], device))
            ans_logits.append(logits[0])
            prop_logits.append(logits[1])
            entity_logits.append(logits[2])
        # 以下两种融合方式二选一
        # 分类模型平均融合
        ans_labels, prop_labels, entity_labels = average_integration(ans_logits, prop_logits, entity_logits, label_hub)
        # # 分类模型投票融合:效果不好，已被弃用
        # ans_labels, prop_labels, entity_labels = vote_integration(ans_logits, prop_logits, entity_logits, label_hub, args.batch_size)
        # ner标签获取
        questions = process_syn(questions, entity_labels, ans_labels)
        test_word_lists = process_data(questions)
        test_tag_lists = [[] for _ in range(len(test_word_lists))]
        # kfold 决策方法
        # ner_logits_list = []
        # for i in range(args.kfold):
        #     ner_logits, _ = ner_model_list[i].test(test_word_lists, test_tag_lists, word2id_list[i], tag2id_list[i])
        #     ner_logits_list.append(ner_logits)
        # kfold 平均分方法
        ner_score_list = []
        for i in range(args.kfold):
            ner_score = ner_model_list[i].test_k_pre(test_word_lists, test_tag_lists, word2id_list[i])
            ner_score_list.append(ner_score)
        avg_score = torch.mean(torch.stack(ner_score_list), dim=0)
        ner_logits, _ = ner_model_list[0].test_k_post(test_word_lists, test_tag_lists, word2id_list[0], tag2id_list[0], avg_score)
        # 获得答案
        for i, question in enumerate(questions):
            # 处理ner结果，得到 main_property 和 sub_properties
            question = question.replace(' ', '_')
            # ner 5-fold 平均值方式
            pre_dict = get_anno_dict(questions[i], ner_logits[i], service_names)
            # ner 5-fold 预测结果 并统计次数
            # temp_dict = collections.defaultdict(lambda: 0)
            # for fold in range(args.kfold):
            #     pre_dict = get_anno_dict(questions[i], ner_logits_list[fold][i], service_names)
            #     for name, values in pre_dict.items():
            #         for value in values:
            #             temp_dict[name + '_' + value] += 1
            # # ner 5-fold 决策 加入次数大于2的，去掉次数小于等于2的
            # for n, v in temp_dict.items():
            #     if v < 5:
            #         name, value = n.split('_')
            #         if v > 2:
            #             if name not in pre_dict:
            #                 pre_dict[name] = [value]
            #             else:
            #                 if value not in pre_dict[name]:
            #                     pre_dict[name].append(value)
            #         else:
            #             if name in pre_dict and value in pre_dict[name]:
            #                 pre_dict[name].remove(value)
            #                 if len(pre_dict[name]) == 0:
            #                     pre_dict.pop(name)
            #         print(n, v)
            #         print(question)
            #         print(pre_dict)
            # 整理ner的预测结果格式
            sub_properties = []
            for k, v in pre_dict.items():
                for vi in range(len(v)):
                    sub_properties.append((k, v[vi]))
            # 过滤 sub_properties
            sub_properties_filter = filter_sub_properties(sub_properties, entity_labels[i], ans_labels[i])
            # todo 使用规则方法进行后处理
            if ans_labels[i] == '比较句':
                if len(sub_properties_filter) == 3:
                    sub_properties_filter = sub_properties_filter[1:]
            sub_properties_filter = process_postdo(sub_properties_filter, question, entity_labels[i])
            # 得到 operator
            opr, obj = get_operator(question)
            if opr is None:
                operator = 'other'
            else:
                operator = opr
                sub_properties_filter = [(obj, 1)]
            # 使用规则来判断"属性名"，判断成功则直接覆盖模型预测结果
            prop_label = rules_to_judge_prop(question)
            if prop_label is not None:
                prop_labels[i] = [prop_label]
            # 考虑到模型对开通和取消在某些场景下会混乱，是否是“取消”类型的句子
            if judge_cancel(question):
                cancel_count = open_count = 0
                for prop in prop_labels[i]:
                    if '开通' in prop:
                        open_count += 1
                    elif '取消' in prop:
                        cancel_count += 1
                if cancel_count == 0 and open_count == 1:
                    for jdx in range(len(prop_labels[i])):
                        prop_labels[i][jdx] = prop_labels[i][jdx].replace('开通', '取消')
            # ---------使用模型来判断开通方式还是开通条件------------
            if len(binary_model_list) > 0:
                if '档位介绍-开通方式' in prop_labels[i] and '档位介绍-开通条件' not in prop_labels[i]:
                    logits = []
                    for j, binary_model in enumerate(binary_model_list):
                        logit = binary_model(**tokenize([question], binary_tokenizer_list[j], device))
                        logits.append(logit)
                    # 平均融合
                    label = binary_average_integration(logits, label_hub)
                    if label[0] != '档位介绍-开通方式':
                        print('------------{}:开通方式遭替换----------------'.format(question))
                    prop_labels[i].remove('档位介绍-开通方式')
                    prop_labels[i] += label
                elif '档位介绍-开通方式' not in prop_labels[i] and '档位介绍-开通条件' in prop_labels[i]:
                    logits = []
                    for j, binary_model in enumerate(binary_model_list):
                        logit = binary_model(**tokenize([question], binary_tokenizer_list[j], device))
                        logits.append(logit)
                    # 平均融合
                    label = binary_average_integration(logits, label_hub)
                    if label[0] != '档位介绍-开通条件':
                        print('------------{}:开通条件遭替换----------------'.format(question))
                    prop_labels[i].remove('档位介绍-开通条件')
                    prop_labels[i] += label
            entity_labels[i], prop_labels[i] = process_postdo_last(question, entity_labels[i], prop_labels[i])
            # 模型预测的结果
            model_result = {
                'question': question,
                'ans_type': ans_labels[i],
                'entity': entity_labels[i],
                'main_property': prop_labels[i],
                'operator': operator,
                'sub_properties': sub_properties_filter}
            temp_ans = kg.fetch_ans(**model_result)
            if len(temp_ans) == 0:
                # 1 答案为空且出现意思相近的实体时
                changed, entity_label = reverse_entity(entity_labels[i])
                if changed:
                    ori_entity = model_result['entity']
                    model_result['entity'] = entity_label
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['entity'] = ori_entity
                # 2 答案为空且出现意思相近的属性名时
                if (len(temp_ans) == 0 or (len(temp_ans)==1 and ans_labels[i] == '并列句')) and reverse_prop(prop_labels[i]):
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = prop_labels[i]
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
                # 3 答案为空且约束属性名不为空时，尽最大可能找答案
                if len(temp_ans) == 0 and len(sub_properties_filter) > 0:
                    model_result['sub_properties'] = []
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['sub_properties'] = sub_properties_filter
                # 4 属性名换成“档位介绍-业务简介”
                if len(temp_ans) == 0:
                    ori_prop = model_result['main_property']
                    model_result['main_property'] = ['档位介绍-业务简介']
                    temp_ans = kg.fetch_ans(**model_result)
                    model_result['main_property'] = ori_prop
            if len(temp_ans) == 0:
                empty_ans.append(model_result)
            for j in range(len(temp_ans)):
                temp_ans[j] = str(temp_ans[j])
            ans['model_result'][str(len(ans['model_result']))] = model_result
            ans['result'][str(len(ans['result']))] = '|'.join(temp_ans)
    # 读取真实结果，计算得分
    if args.dev == '1':
        with open('../data/dataset/cls_dev.txt', 'r', encoding='utf-8') as f:
            answer = {}
            for i, line in enumerate(f):
                one_ans = line.strip().split('\t')[-2]
                answer[str(i)] = one_ans
        # print('\n验证结果是：', score_evalution(answers=answer, predictions=ans['result']))
        with open('../data/results/ensemble_{}_dev.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)
    # 直接写入答案
    else:
        if args.debug == '1':
            with open('../data/results/ensemble_{}.json'.format(args.cls_model.replace(':', '：')), 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
            generate_false_label('ensemble_{}.json'.format(args.cls_model.replace(':', '：')))
        else:
            with open('/code/result.json', 'w', encoding='utf-8') as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
    print('答案写入成功，空结果有{}个。'.format(len(empty_ans)))
    for e_ans in empty_ans:
        print(e_ans)

    # 查看预测的效果
    print()
    for id, detail in ans['model_result'].items():
        print(id, end='=')
        print(detail['question'], end='=')
        print(detail['ans_type'], end='=')
        print(detail['entity'], end='=')
        for i in range(len(detail['main_property'])):
            if i < len(detail['main_property']) - 1:
                print(detail['main_property'][i], end='x')
            else:
                print(detail['main_property'][i], end='=')
        if len(detail['main_property']) == 0:
            print(end='=')
        print(detail['operator'], end='=')
        for i in range(len(detail['sub_properties'])):
            if i < len(detail['sub_properties']) - 1:
                print(str(detail['sub_properties'][i][0]) + '_' + str(detail['sub_properties'][i][1]), end='x')
            else:
                print(str(detail['sub_properties'][i][0]) + '_' + str(detail['sub_properties'][i][1]), end='+')
        if len(detail['sub_properties']) == 0:
            print(end='+')
    print()


if __name__ == '__main__':
    print('-----------------------开始inference--------------------------------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', '-c', default=0, type=int)
    parser.add_argument('--debug', '-de', default='0', type=str, choices=['0', '1'])
    parser.add_argument('--batch_size', '-b', default=4, type=int)
    parser.add_argument('--model_path', '-mp', default='../data/trained_model/')
    parser.add_argument('--data_path', '-dp', default='../data/dataset/cls_unlabeled.txt')
    parser.add_argument('--tag_version', '-tv', default='bieo', type=str)

    # parser.add_argument('--cls_model', '-cm', required=True, help='cls model name')
    parser.add_argument('--cls_model', '-cm', default='bert_model.pth', help='cls model name')
    parser.add_argument('--xlnet_model', '-xm', default='', help='cls model name')
    parser.add_argument('--electra_model', '-em', default='', help='cls model name')
    parser.add_argument('--roberta_model', '-rm', default='', help='cls model name')
    parser.add_argument('--albert_model', '-am', default='', help='cls model name')
    parser.add_argument('--gpt2_model', '-gm', default='', help='cls model name')
    parser.add_argument('--macbert_model', '-mm', default='', help='cls model name')

    # parser.add_argument('--ner_model', '-nm', required=True, help='ner model name')
    parser.add_argument('--ner_model', '-nm', default='ner_model.pth', help='ner model name')
    parser.add_argument('--dev', '-d', default='0', type=str)
    parser.add_argument('--ensemble', '-e', default='0', type=str)
    parser.add_argument('--fold', '-f', action='store_true', default=False, help='是否加入5折交叉验证的结果')
    parser.add_argument('--kfold', '-k', default=5, type=int)
    parser.add_argument('--with_binary', '-wb', action='store_true', default=False, help='是否加入二分类模型修正开通条件、方式')
    args = parser.parse_args()
    if args.ensemble == '0':
        if args.kfold == 5:
            predict_kfold()
        else:
            predict()
    else:
        if args.with_binary:
            predict_ensemble_with_binary()
        else:
            predict_ensemble()
    print('-----------------------docker运行结束，时间为{}-----------------------'.format(get_time_str()))
