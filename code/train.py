"""
@Author: hezf 
@Time: 2021/6/3 14:42 
@desc: 
"""
from model import PTModel, PTModelForMultiClassification, PTModelForProperty, PTModelAttention, PTModelTaskAttention, PTModelForPropertyV2, PTModelTaskAttentionV1, PTModelBiClassifier
from data import bert_collate_fn, BertDataset, LabelHub, BinaryDataset, binary_collate_fn
from torch.utils.data import random_split, DataLoader
import argparse
import torch
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AutoModel, AutoTokenizer, AutoConfig
import os
from module import MutiTaskLoss, Metric, RDropLoss, MutiTaskLossV1, CrossEntropyWithEfficiency
from utils import (
    logits_to_multi_hot, get_time_dif, view_gpu_info, load_model,
    setup_seed, save_model,
    get_time_str, logits_to_multi_hot_old_version,
    k_fold_data, MyLogger, tqdm_with_debug)
from sklearn.metrics import f1_score, jaccard_score

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# '/data2/hezhenfeng/drug_recommendation/data/model/bert_pretrained_model'
# '/data/x/temp/drug_recommendation/data/model/bert_pretrained_model'
BERT_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/bert-base-chinese'
XLNET_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-xlnet-base'
ROBERTA_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-roberta-wwm-ext'
ELECTRA_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-electra-180g-base-discriminator'
ALBERT_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/albert-chinese-tiny'
GPT2_SMALL_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/gpt2-chinese-small'
GPT2_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/gpt2-chinese-base'
MACBERT_PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/chinese-macbert-base'
PRETRAINED_PATH = '/data2/hezhenfeng/huggingface-model/bert-base-chinese'


def train():
    # log
    start_time = get_time_str()
    model_name = '{}_{}_seed{}_gpu{}.pth'.format(args.model_name, start_time, args.seed, args.gpu_type)
    logger = MyLogger(log_file=log_path + 'time{} model_name{}.log'.format(start_time, model_name),
                      debug=args.debug)
    # data
    batch_size = args.batch_size
    if 'albert-chinese-tiny' in PRETRAINED_PATH:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    label_hub = LabelHub('../data/dataset/{}'.format(args.label_file))
    train_dataset = BertDataset('../data/dataset/cls_train.txt', tokenizer, label_hub)
    dev_dataset = BertDataset('../data/dataset/cls_dev.txt', tokenizer, label_hub)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    # logger.log('当前可见的卡：', os.environ['CUDA_VISIBLE_DEVICES'])
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    pt_model = AutoModel.from_pretrained(PRETRAINED_PATH, config=config)
    if args.trained_model != '':
        model = load_model(os.path.join('../data/trained_model/', args.trained_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device)
    else:
        model = PTModelTaskAttention(model=pt_model,
                                     ans_class=len(label_hub.ans_label2id),
                                     prop_label=len(label_hub.prop_label2id),
                                     entity_class=len(label_hub.entity_label2id),
                                     model_config=config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[6, 15, 40, 60, 100], gamma=0.5)
    loss_func = MutiTaskLoss(use_efficiency=True).to(device)
    metric = Metric()
    # train
    max_metric = best_epoch = 0
    logger.log('model:\n{}'.format(model))
    for i in range(args.epoch):
        model.train()
        train_loss, dev_loss, ans_f1_train, ans_f1_dev = [], [], [], []
        prop_j_train, prop_j_dev, entity_f1_train, entity_f1_dev = [], [], [], []
        dev_pred_list, dev_gold_list = [], []
        logger.log('epoch {}'.format(i))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        logger.log('time : {}'.format(get_time_str()))
        for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(train_dataloader, debug=args.debug):
            input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
            ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
            efficiency = efficiency.to(device)
            ans_logits, prop_logits, entity_logits = model(input_ids=input_ids,
                                                           token_type_ids=token_type_ids,
                                                           attention_mask=attention_mask)
            optimizer.zero_grad()
            loss = loss_func(ans_labels, ans_logits, prop_labels, prop_logits, entity_labels, entity_logits, efficiency)
            train_loss.append(loss.item())
            # results
            ans_pred = ans_logits.data.cpu().argmax(dim=1)
            prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
            entity_pred = entity_logits.data.cpu().argmax(dim=1)
            # socre
            ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                         prop_pred, prop_labels.data.cpu(),
                                                         entity_pred, entity_labels.data.cpu())
            ans_f1_train.append(ans_f1)
            prop_j_train.append(prop_j)
            entity_f1_train.append(entity_f1)
            # optim
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.log('train_loss:{}'.format(np.mean(train_loss)))
        logger.log('ans_f1_train:{}'.format(np.mean(ans_f1_train)))
        logger.log('prop_j_train:{}'.format(np.mean(prop_j_train)))
        logger.log('entity_f1_train:{}'.format(np.mean(entity_f1_train)))
        with torch.no_grad():
            model.eval()
            for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(dev_dataloader,debug=args.debug):
                input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
                efficiency = efficiency.to(device)
                ans_logits, prop_logits, entity_logits = model(input_ids=input_ids,
                                                               token_type_ids=token_type_ids,
                                                               attention_mask=attention_mask)
                loss = loss_func(ans_labels, ans_logits, prop_labels, prop_logits, entity_labels, entity_logits, efficiency)
                dev_loss.append(loss.item())
                # result
                ans_pred = ans_logits.data.cpu().argmax(dim=1)
                dev_pred_list += ans_pred
                dev_gold_list += list(np.array(ans_labels.data.cpu()))
                prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
                entity_pred = entity_logits.data.cpu().argmax(dim=1)
                # score
                ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                             prop_pred, prop_labels.data.cpu(),
                                                             entity_pred, entity_labels.data.cpu())
                ans_f1_dev.append(ans_f1)
                prop_j_dev.append(prop_j)
                entity_f1_dev.append(entity_f1)
            avg_ans_f1 = np.mean(ans_f1_dev)
            avg_prop_j = np.mean(prop_j_dev)
            avg_entity_f1 = np.mean(entity_f1_dev)
            logger.log('dev_loss:{}'.format(np.mean(dev_loss)))
            logger.log('ans_f1_dev:{}'.format(avg_ans_f1))
            logger.log('prop_j_dev:{}'.format(avg_prop_j))
            logger.log('entity_f1_dev:{}'.format(avg_entity_f1))
            logger.log('dev_ans_confusion_matrix:\n {}'.format(confusion_matrix(dev_gold_list, dev_pred_list)))
            cur_metric = avg_ans_f1 * avg_prop_j * avg_entity_f1
            if cur_metric > max_metric:
                best_epoch = i
                max_metric = cur_metric
                if args.debug == '1':
                    save_model(model=model,
                               model_path='../data/trained_model/',
                               model_name='{}_{}_seed{}_gpu{}_be{}_{:.8f}.pth'
                               .format(args.model_name, start_time, args.seed, args.gpu_type, best_epoch, max_metric),
                               debug=args.debug)
                else:
                    save_model(model=model,
                               model_path='../data/trained_model/',
                               model_name='{}.pth'
                               .format(args.model_name),
                               debug=args.debug)
            logger.log('current result: {}'.format(cur_metric))
            logger.log('best result: {}'.format(max_metric))
            logger.log('best epoch: {}\n\n'.format(best_epoch))


def k_fold_train():
    # log
    start_time = get_time_str()
    model_name = '{}_{}_seed{}_gpu{}.pth'.format(args.model_name, start_time, args.seed, args.gpu_type)
    logger = MyLogger(log_file=log_path + 'time{} {}.log'.format(start_time, model_name),
                      debug=args.debug)
    # data
    batch_size = args.batch_size
    if 'albert-chinese-tiny' in PRETRAINED_PATH:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    label_hub = LabelHub('../data/dataset/{}'.format(args.label_file))
    dataset = BertDataset('../data/dataset/{}'.format(args.train_file), tokenizer, label_hub)
    # 默认第一组来训练
    train_dev_tuples = k_fold_data(dataset.data, k=args.fold, batch_size=batch_size, seed=args.seed)[args.fold_id:args.fold_id+1]
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    # logger.log('当前可见的卡：', os.environ['CUDA_VISIBLE_DEVICES'])
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    pt_model = AutoModel.from_pretrained(PRETRAINED_PATH, config=config)
    if args.trained_model != '':
        model = load_model(os.path.join('../data/trained_model/', args.trained_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device)
    else:
        model = PTModelTaskAttention(model=pt_model,
                                     ans_class=len(label_hub.ans_label2id),
                                     prop_label=len(label_hub.prop_label2id),
                                     entity_class=len(label_hub.entity_label2id),
                                     model_config=config,
                                     dropout_p=0.1).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 40, 60, 100], gamma=0.5)
    loss_func = MutiTaskLoss(use_efficiency=True).to(device)
    metric = Metric()
    # train
    max_metric = best_epoch = 0
    logger.log('model:\n{}'.format(model))
    for i in range(args.epoch):
        model.train()
        logger.log('epoch {}'.format(i))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        logger.log('time : {}'.format(get_time_str()))
        if i == 1:
            view_gpu_info(args.cuda)
        temp_best_values = []
        for _, (train_dataloader, dev_dataloader) in enumerate(train_dev_tuples):
            logger.log('fold {}'.format(_))
            train_loss, dev_loss, ans_f1_train, ans_f1_dev = [], [], [], []
            prop_j_train, prop_j_dev, entity_f1_train, entity_f1_dev = [], [], [], []
            dev_pred_list, dev_gold_list = [], []
            temp_best = 0
            for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(train_dataloader, debug=args.debug):
                input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
                efficiency = efficiency.to(device)
                ans_logits, prop_logits, entity_logits = model(input_ids=input_ids,
                                                               token_type_ids=token_type_ids,
                                                               attention_mask=attention_mask)
                optimizer.zero_grad()
                loss = loss_func(ans_labels, ans_logits, prop_labels, prop_logits, entity_labels, entity_logits, efficiency)
                train_loss.append(loss.item())
                # results
                ans_pred = ans_logits.data.cpu().argmax(dim=1)
                prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
                entity_pred = entity_logits.data.cpu().argmax(dim=1)
                # socre
                ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                             prop_pred, prop_labels.data.cpu(),
                                                             entity_pred, entity_labels.data.cpu())
                ans_f1_train.append(ans_f1)
                prop_j_train.append(prop_j)
                entity_f1_train.append(entity_f1)
                # optim
                loss.backward()
                optimizer.step()
            scheduler.step()
            logger.log('train_loss:{}'.format(np.mean(train_loss)))
            logger.log('ans_f1_train:{}'.format(np.mean(ans_f1_train)))
            logger.log('prop_j_train:{}'.format(np.mean(prop_j_train)))
            logger.log('entity_f1_train:{}'.format(np.mean(entity_f1_train)))
            with torch.no_grad():
                model.eval()
                for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(dev_dataloader, debug=args.debug):
                    input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                    ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
                    efficiency = efficiency.to(device)
                    ans_logits, prop_logits, entity_logits = model(input_ids=input_ids,
                                                                   token_type_ids=token_type_ids,
                                                                   attention_mask=attention_mask)
                    loss = loss_func(ans_labels, ans_logits, prop_labels, prop_logits, entity_labels, entity_logits, efficiency)
                    dev_loss.append(loss.item())
                    # result
                    ans_pred = ans_logits.data.cpu().argmax(dim=1)
                    dev_pred_list += ans_pred
                    dev_gold_list += list(np.array(ans_labels.data.cpu()))
                    prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
                    entity_pred = entity_logits.data.cpu().argmax(dim=1)
                    # score
                    ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                                 prop_pred, prop_labels.data.cpu(),
                                                                 entity_pred, entity_labels.data.cpu())
                    ans_f1_dev.append(ans_f1)
                    prop_j_dev.append(prop_j)
                    entity_f1_dev.append(entity_f1)
                avg_ans_f1 = np.mean(ans_f1_dev)
                avg_prop_j = np.mean(prop_j_dev)
                avg_entity_f1 = np.mean(entity_f1_dev)
                logger.log('dev_loss:{}'.format(np.mean(dev_loss)))
                logger.log('ans_f1_dev:{}'.format(avg_ans_f1))
                logger.log('prop_j_dev:{}'.format(avg_prop_j))
                logger.log('entity_f1_dev:{}'.format(avg_entity_f1))
                logger.log('dev_ans_confusion_matrix:\n {}'.format(confusion_matrix(dev_gold_list, dev_pred_list)))
                if temp_best < avg_ans_f1 * avg_prop_j * avg_entity_f1:
                    temp_best = avg_ans_f1 * avg_prop_j * avg_entity_f1
        temp_best_values.append(temp_best)
        if np.mean(temp_best_values) > max_metric:
            best_epoch = i
            max_metric = np.mean(temp_best_values)
            if args.debug == '1':
                save_model(model=model,
                           model_path='../data/trained_model/',
                           model_name='{}_{}_seed{}_fi{}_gpu{}_be{}_{:.8f}.pth'
                           .format(args.model_name, start_time, args.seed, args.fold_id, args.gpu_type, best_epoch, max_metric),
                           debug=args.debug)
            else:
                save_model(model=model,
                           model_path='../data/trained_model/',
                           model_name='{}.pth'
                           .format(args.model_name),
                           debug=args.debug)

        print('current result: {}'.format(np.mean(temp_best_values)))
        print('best result: {}'.format(max_metric))
        print('best epoch: {}'.format(best_epoch))
        logger.log('current result: {}'.format(np.mean(temp_best_values)))
        logger.log('best result: {}'.format(max_metric))
        logger.log('best epoch: {}\n\n\n'.format(best_epoch))


def k_fold_train_with_cm():
    # log
    start_time = get_time_str()
    model_name = '{}_{}_seed{}_gpu{}.pth'.format(args.model_name, start_time, args.seed, args.gpu_type)
    logger = MyLogger(log_file=log_path + 'time{} {}.log'.format(start_time, model_name),
                      debug=args.debug)
    # data
    batch_size = args.batch_size
    if 'albert-chinese-tiny' in PRETRAINED_PATH:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    label_hub = LabelHub('../data/dataset/{}'.format(args.label_file))
    dataset = BertDataset('../data/dataset/{}'.format(args.train_file), tokenizer, label_hub)
    # 默认第一组来训练
    train_dev_tuples = k_fold_data(dataset.data, k=args.fold, batch_size=batch_size, seed=args.seed)[args.fold_id:args.fold_id+1]
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    # logger.log('当前可见的卡：', os.environ['CUDA_VISIBLE_DEVICES'])
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    pt_model = AutoModel.from_pretrained(PRETRAINED_PATH, config=config)
    if args.trained_model != '':
        model = load_model(os.path.join('../data/trained_model/', args.trained_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device)
    else:
        model = PTModelTaskAttentionV1(model=pt_model,
                                       ans_class=len(label_hub.ans_label2id),
                                       prop_label=len(label_hub.prop_label2id),
                                       entity_class=len(label_hub.entity_label2id),
                                       model_config=config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 40, 60, 100], gamma=0.5)
    loss_func = MutiTaskLossV1(use_efficiency=True).to(device)
    metric = Metric()
    # train
    max_metric = best_epoch = 0
    logger.log('model:\n{}'.format(model))
    m_id, c_id = label_hub.prop_label2id['档位介绍-开通方式'], label_hub.prop_label2id['档位介绍-开通条件']
    for i in range(args.epoch):
        model.train()
        logger.log('epoch {}'.format(i))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        logger.log('time : {}'.format(get_time_str()))
        if i == 1:
            view_gpu_info(args.cuda)
        temp_best_values = []
        for _, (train_dataloader, dev_dataloader) in enumerate(train_dev_tuples):
            logger.log('fold {}'.format(_))
            train_loss, dev_loss, ans_f1_train, ans_f1_dev = [], [], [], []
            prop_j_train, prop_j_dev, entity_f1_train, entity_f1_dev = [], [], [], []
            me_f1_train, me_f1_dev, cond_f1_train, cond_f1_dev = [], [], [], []
            dev_pred_list, dev_gold_list = [], []
            temp_best = 0
            for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(train_dataloader, debug=args.debug):
                input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
                method_labels, condition_labels = prop_labels[:, [m_id]].squeeze(1), prop_labels[:, [c_id]].squeeze(1)
                method_labels, condition_labels = method_labels.to(device), condition_labels.to(device)
                efficiency = efficiency.to(device)
                ans_logits, prop_logits, entity_logits, method_logits, condition_logits = model(input_ids=input_ids,
                                                                                                token_type_ids=token_type_ids,
                                                                                                attention_mask=attention_mask)
                optimizer.zero_grad()
                loss = loss_func(ans_labels, ans_logits, prop_labels, prop_logits, entity_labels, entity_logits,
                                 method_labels, method_logits, condition_labels, condition_logits, efficiency)
                train_loss.append(loss.item())
                # results
                ans_pred = ans_logits.data.cpu().argmax(dim=1)
                prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
                entity_pred = entity_logits.data.cpu().argmax(dim=1)
                method_pred = method_logits.data.cpu().argmax(dim=1)
                condition_pred = condition_logits.data.cpu().argmax(dim=1)
                # socre
                ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                             prop_pred, prop_labels.data.cpu(),
                                                             entity_pred, entity_labels.data.cpu())
                method_f1, condition_f1 = f1_score(method_labels.data.cpu(), method_pred, average='micro'), f1_score(condition_labels.data.cpu(), condition_pred, average='micro')
                ans_f1_train.append(ans_f1)
                prop_j_train.append(prop_j)
                entity_f1_train.append(entity_f1)
                me_f1_train.append(method_f1)
                cond_f1_train.append(condition_f1)
                # optim
                loss.backward()
                optimizer.step()
            scheduler.step()
            logger.log('train_loss:{}'.format(np.mean(train_loss)))
            logger.log('ans_f1_train:{}'.format(np.mean(ans_f1_train)))
            logger.log('prop_j_train:{}'.format(np.mean(prop_j_train)))
            logger.log('entity_f1_train:{}'.format(np.mean(entity_f1_train)))
            logger.log('method_f1_train:{}'.format(np.mean(me_f1_train)))
            logger.log('condition_f1_train:{}'.format(np.mean(cond_f1_train)))
            with torch.no_grad():
                model.eval()
                for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(dev_dataloader, debug=args.debug):
                    input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                    ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
                    method_labels, condition_labels = prop_labels[:, [m_id]].squeeze(1), prop_labels[:, [c_id]].squeeze(1)
                    method_labels, condition_labels = method_labels.to(device), condition_labels.to(device)
                    efficiency = efficiency.to(device)
                    ans_logits, prop_logits, entity_logits, method_logits, condition_logits = model(input_ids=input_ids,
                                                                                                    token_type_ids=token_type_ids,
                                                                                                    attention_mask=attention_mask)
                    loss = loss_func(ans_labels, ans_logits, prop_labels, prop_logits, entity_labels, entity_logits,
                                     method_labels, method_logits, condition_labels, condition_logits, efficiency)
                    dev_loss.append(loss.item())
                    # result
                    ans_pred = ans_logits.data.cpu().argmax(dim=1)
                    dev_pred_list += ans_pred
                    dev_gold_list += list(np.array(ans_labels.data.cpu()))
                    prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
                    entity_pred = entity_logits.data.cpu().argmax(dim=1)
                    method_pred = method_logits.data.cpu().argmax(dim=1)
                    condition_pred = condition_logits.data.cpu().argmax(dim=1)
                    # score
                    ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                                 prop_pred, prop_labels.data.cpu(),
                                                                 entity_pred, entity_labels.data.cpu())
                    method_f1, condition_f1 = f1_score(method_labels.data.cpu(), method_pred, average='micro'), f1_score(condition_labels.data.cpu(), condition_pred, average='micro')
                    ans_f1_dev.append(ans_f1)
                    prop_j_dev.append(prop_j)
                    entity_f1_dev.append(entity_f1)
                    me_f1_dev.append(method_f1)
                    cond_f1_dev.append(condition_f1)
                avg_ans_f1 = np.mean(ans_f1_dev)
                avg_prop_j = np.mean(prop_j_dev)
                avg_entity_f1 = np.mean(entity_f1_dev)
                logger.log('dev_loss:{}'.format(np.mean(dev_loss)))
                logger.log('ans_f1_dev:{}'.format(avg_ans_f1))
                logger.log('prop_j_dev:{}'.format(avg_prop_j))
                logger.log('entity_f1_dev:{}'.format(avg_entity_f1))
                logger.log('method_f1_dev:{}'.format(np.mean(me_f1_dev)))
                logger.log('condition_f1_dev:{}'.format(np.mean(cond_f1_dev)))
                logger.log('dev_ans_confusion_matrix:\n {}'.format(confusion_matrix(dev_gold_list, dev_pred_list)))
                if temp_best < avg_ans_f1 * avg_prop_j * avg_entity_f1:
                    temp_best = avg_ans_f1 * avg_prop_j * avg_entity_f1
        temp_best_values.append(temp_best)
        if np.mean(temp_best_values) > max_metric:
            best_epoch = i
            max_metric = np.mean(temp_best_values)
            if args.debug == '1':
                save_model(model=model,
                           model_path='../data/trained_model/',
                           model_name='{}_{}_seed{}_fi{}_gpu{}_be{}_{:.8f}.pth'
                           .format(args.model_name, start_time, args.seed, args.fold_id, args.gpu_type, best_epoch, max_metric),
                           debug=args.debug)
            else:
                save_model(model=model,
                           model_path='../data/trained_model/',
                           model_name='{}.pth'
                           .format(args.model_name),
                           debug=args.debug)

        print('current result: {}'.format(np.mean(temp_best_values)))
        print('best result: {}'.format(max_metric))
        print('best epoch: {}'.format(best_epoch))
        logger.log('current result: {}'.format(np.mean(temp_best_values)))
        logger.log('best result: {}'.format(max_metric))
        logger.log('best epoch: {}\n\n\n'.format(best_epoch))


def train_without_dev():
    # log
    start_time = get_time_str()
    model_name = 'pretrained_{}_{}_seed{}_gpu{}.pth'.format(args.model_name, start_time, args.seed, args.gpu_type)
    logger = MyLogger(log_file=log_path + 'time{} {}.log'.format(start_time, model_name), debug=args.debug)
    # data
    batch_size = args.batch_size
    if 'albert-chinese-tiny' in PRETRAINED_PATH:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    label_hub = LabelHub('../data/dataset/{}'.format(args.label_file))
    train_dataset = BertDataset('../data/dataset/{}'.format(args.train_file), tokenizer, label_hub)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    pt_model = AutoModel.from_pretrained(PRETRAINED_PATH, config=config)
    model = PTModelTaskAttention(model=pt_model,
                                 ans_class=len(label_hub.ans_label2id),
                                 prop_label=len(label_hub.prop_label2id),
                                 entity_class=len(label_hub.entity_label2id),
                                 model_config=config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 40, 60, 100], gamma=0.5)
    loss_func = MutiTaskLoss(use_efficiency=True).to(device)
    metric = Metric()
    # train
    max_metric = best_epoch = 0
    logger.log('model:\n{}'.format(model))
    for i in range(args.epoch):
        model.train()
        logger.log('epoch {}'.format(i))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        logger.log('time : {}'.format(get_time_str()))
        if i == 1:
            view_gpu_info(args.cuda)
        train_loss, dev_loss, ans_f1_train, ans_f1_dev = [], [], [], []
        prop_j_train, prop_j_dev, entity_f1_train, entity_f1_dev = [], [], [], []
        for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(train_dataloader, debug=args.debug):
            input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
            ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
            efficiency = efficiency.to(device)
            ans_logits, prop_logits, entity_logits = model(input_ids=input_ids,
                                                           token_type_ids=token_type_ids,
                                                           attention_mask=attention_mask)
            optimizer.zero_grad()
            loss = loss_func(ans_labels, ans_logits, prop_labels, prop_logits, entity_labels, entity_logits, efficiency)
            train_loss.append(loss.item())
            # results
            ans_pred = ans_logits.data.cpu().argmax(dim=1)
            prop_pred = logits_to_multi_hot(prop_logits, ans_pred, label_hub)
            entity_pred = entity_logits.data.cpu().argmax(dim=1)
            # socre
            ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                         prop_pred, prop_labels.data.cpu(),
                                                         entity_pred, entity_labels.data.cpu())
            ans_f1_train.append(ans_f1)
            prop_j_train.append(prop_j)
            entity_f1_train.append(entity_f1)
            # optim
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('ans_f1: {}  prop_j:{} entity_f1:{}'.format(np.mean(ans_f1_train), np.mean(prop_j_train), np.mean(entity_f1_train)))
        logger.log('train_loss:{}'.format(np.mean(train_loss)))
        logger.log('ans_f1_train:{}'.format(np.mean(ans_f1_train)))
        logger.log('prop_j_train:{}'.format(np.mean(prop_j_train)))
        logger.log('entity_f1_train:{}'.format(np.mean(entity_f1_train)))
    if args.debug == '1':
        save_model(model=model,
                   model_path='../data/trained_model/',
                   model_name='pretrained_{}_{}_seed{}_fi{}_gpu{}_be{}_{:.8f}.pth'
                   .format(args.model_name, start_time, args.seed, args.fold_id, args.gpu_type, best_epoch, max_metric),
                   debug=args.debug)
    else:
        save_model(model=model,
                   model_path='../data/trained_model/',
                   model_name='pretrained_{}.pth'
                   .format(args.model_name),
                   debug=args.debug)


def r_drop_train():
    # log
    start_time = get_time_str()
    model_name = '{}_{}_seed{}_gpu{}.pth'.format(args.model_name, start_time, args.seed, args.gpu_type)
    logger = MyLogger(log_file=log_path + 'time{} {}.log'.format(start_time, model_name),
                      debug=args.debug)
    # data
    batch_size = args.batch_size
    if 'albert-chinese-tiny' in PRETRAINED_PATH:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    label_hub = LabelHub('../data/dataset/{}'.format(args.label_file))
    dataset = BertDataset('../data/dataset/{}'.format(args.train_file), tokenizer, label_hub)
    # 默认第一组来训练
    train_dev_tuples = k_fold_data(dataset.data, k=args.fold, batch_size=batch_size, seed=args.seed)[args.fold_id:args.fold_id + 1]
    train_dataloader, dev_dataloader = train_dev_tuples[0]
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    # logger.log('当前可见的卡：', os.environ['CUDA_VISIBLE_DEVICES'])
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    pt_model = AutoModel.from_pretrained(PRETRAINED_PATH, config=config)
    model = PTModelTaskAttention(model=pt_model,
                                 ans_class=len(label_hub.ans_label2id),
                                 prop_label=len(label_hub.prop_label2id),
                                 entity_class=len(label_hub.entity_label2id),
                                 model_config=config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 40, 60, 100], gamma=0.5)
    loss_func = RDropLoss(alpha=1).to(device)
    metric = Metric()
    # train
    max_metric = best_epoch = 0
    logger.log('model:\n{}'.format(model))
    for i in range(args.epoch):
        model.train()
        train_loss, dev_loss, ans_f1_train, ans_f1_dev = [], [], [], []
        prop_j_train, prop_j_dev, entity_f1_train, entity_f1_dev = [], [], [], []
        dev_pred_list, dev_gold_list = [], []
        logger.log('epoch {}'.format(i))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        logger.log('time : {}'.format(get_time_str()))
        if i == 1:
            view_gpu_info(args.cuda)
        for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(train_dataloader, debug=args.debug):
            input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
            ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
            efficiency = efficiency.to(device)
            ans_logits0, prop_logits0, entity_logits0 = model(input_ids=input_ids,
                                                           token_type_ids=token_type_ids,
                                                           attention_mask=attention_mask)
            ans_logits1, prop_logits1, entity_logits1 = model(input_ids=input_ids,
                                                           token_type_ids=token_type_ids,
                                                           attention_mask=attention_mask)
            optimizer.zero_grad()
            loss = loss_func(ans_labels, [ans_logits0, ans_logits1],
                             prop_labels, [prop_logits0, prop_logits1],
                             entity_labels, [entity_logits0, entity_logits1],
                             efficiency)
            train_loss.append(loss.item())
            # results
            ans_pred = ans_logits0.data.cpu().argmax(dim=1)
            prop_pred = logits_to_multi_hot(prop_logits0, ans_pred, label_hub)
            entity_pred = entity_logits0.data.cpu().argmax(dim=1)
            # socre
            ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                         prop_pred, prop_labels.data.cpu(),
                                                         entity_pred, entity_labels.data.cpu())
            ans_f1_train.append(ans_f1)
            prop_j_train.append(prop_j)
            entity_f1_train.append(entity_f1)
            # optim
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.log('train_loss:{}'.format(np.mean(train_loss)))
        logger.log('ans_f1_train:{}'.format(np.mean(ans_f1_train)))
        logger.log('prop_j_train:{}'.format(np.mean(prop_j_train)))
        logger.log('entity_f1_train:{}'.format(np.mean(entity_f1_train)))
        with torch.no_grad():
            model.eval()
            for input_ids, token_type_ids, attention_mask, ans_labels, prop_labels, entity_labels, efficiency in tqdm_with_debug(dev_dataloader, debug=args.debug):
                input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                ans_labels, prop_labels, entity_labels = ans_labels.to(device), prop_labels.to(device), entity_labels.to(device)
                efficiency = efficiency.to(device)
                ans_logits0, prop_logits0, entity_logits0 = model(input_ids=input_ids,
                                                                  token_type_ids=token_type_ids,
                                                                  attention_mask=attention_mask)
                ans_logits1, prop_logits1, entity_logits1 = model(input_ids=input_ids,
                                                                  token_type_ids=token_type_ids,
                                                                  attention_mask=attention_mask)
                optimizer.zero_grad()
                loss = loss_func(ans_labels, [ans_logits0, ans_logits1],
                                 prop_labels, [prop_logits0, prop_logits1],
                                 entity_labels, [entity_logits0, entity_logits1],
                                 efficiency)
                dev_loss.append(loss.item())
                # results
                ans_pred = ans_logits0.data.cpu().argmax(dim=1)
                prop_pred = logits_to_multi_hot(prop_logits0, ans_pred, label_hub)
                entity_pred = entity_logits0.data.cpu().argmax(dim=1)
                # score
                ans_f1, prop_j, entity_f1 = metric.calculate(ans_pred, ans_labels.data.cpu(),
                                                             prop_pred, prop_labels.data.cpu(),
                                                             entity_pred, entity_labels.data.cpu())
                ans_f1_dev.append(ans_f1)
                prop_j_dev.append(prop_j)
                entity_f1_dev.append(entity_f1)
            avg_ans_f1 = np.mean(ans_f1_dev)
            avg_prop_j = np.mean(prop_j_dev)
            avg_entity_f1 = np.mean(entity_f1_dev)
            logger.log('dev_loss:{}'.format(np.mean(dev_loss)))
            logger.log('ans_f1_dev:{}'.format(avg_ans_f1))
            logger.log('prop_j_dev:{}'.format(avg_prop_j))
            logger.log('entity_f1_dev:{}'.format(avg_entity_f1))
            cur_metric = avg_ans_f1 * avg_prop_j * avg_entity_f1
            if cur_metric > max_metric:
                best_epoch = i
                max_metric = cur_metric
                if args.debug == '1':
                    save_model(model=model,
                               model_path='../data/trained_model/',
                               model_name='{}_{}_seed{}_gpu{}_be{}_{:.8f}.pth'
                               .format(args.model_name, start_time, args.seed, args.gpu_type, best_epoch, max_metric),
                               debug=args.debug)
                else:
                    save_model(model=model,
                               model_path='../data/trained_model/',
                               model_name='{}.pth'
                               .format(args.model_name),
                               debug=args.debug)
            print('current result: {}'.format(np.mean(cur_metric)))
            print('best result: {}'.format(max_metric))
            print('best epoch: {}'.format(best_epoch))
            logger.log('current result: {}'.format(cur_metric))
            logger.log('best result: {}'.format(max_metric))
            logger.log('best epoch: {}\n\n'.format(best_epoch))


def k_fold_train_binary():
    # log
    start_time = get_time_str()
    model_name = '{}_{}_seed{}_gpu{}.pth'.format(args.model_name, start_time, args.seed, args.gpu_type)
    logger = MyLogger(log_file=log_path + 'time{} {}.log'.format(start_time, model_name),
                      debug=args.debug)
    # data
    batch_size = args.batch_size
    if 'albert-chinese-tiny' in PRETRAINED_PATH:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    label_hub = LabelHub('../data/dataset/{}'.format(args.label_file))
    dataset = BinaryDataset('../data/dataset/{}'.format(args.train_file), tokenizer, label_hub)
    # 默认第一组来训练
    train_dev_tuples = k_fold_data(dataset.data, k=args.fold, batch_size=batch_size, seed=args.seed, collate_fn='2')[args.fold_id:args.fold_id+1]
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    # logger.log('当前可见的卡：', os.environ['CUDA_VISIBLE_DEVICES'])
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    pt_model = AutoModel.from_pretrained(PRETRAINED_PATH, config=config)
    if args.trained_model != '':
        model = load_model(os.path.join('../data/trained_model/', args.trained_model), map_location=lambda storage, loc: storage.cuda(args.cuda)).to(device)
    else:
        model = PTModelBiClassifier(model=pt_model,
                                    model_config=config,
                                    dropout_p=0.3).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 20], gamma=0.5)
    loss_func = CrossEntropyWithEfficiency().to(device)
    # train
    max_metric = best_epoch = 0
    logger.log('model:\n{}'.format(model))
    for i in range(args.epoch):
        model.train()
        logger.log('epoch {}'.format(i))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        logger.log('time : {}'.format(get_time_str()))
        if i == 1:
            view_gpu_info(args.cuda)
        temp_best_values = []
        for _, (train_dataloader, dev_dataloader) in enumerate(train_dev_tuples):
            logger.log('fold {}'.format(_))
            train_loss, dev_loss, f1_train, f1_dev = [], [], [], []
            dev_pred_list, dev_gold_list = [], []
            temp_best = 0
            for input_ids, token_type_ids, attention_mask, labels, efficiency in tqdm_with_debug(train_dataloader, debug=args.debug):
                input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                labels, efficiency = labels.to(device), efficiency.to(device)
                logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

                optimizer.zero_grad()
                loss = loss_func(labels, logits, efficiency)
                train_loss.append(loss.item())
                # results
                pred = logits.data.cpu().argmax(dim=1)
                # socre
                f1 = f1_score(labels.data.cpu(), pred, average='micro')
                f1_train.append(f1)
                # optim
                loss.backward()
                optimizer.step()
            scheduler.step()
            logger.log('train_loss:{}'.format(np.mean(train_loss)))
            logger.log('f1_train:{}'.format(np.mean(f1_train)))
            with torch.no_grad():
                model.eval()
                for input_ids, token_type_ids, attention_mask, labels, efficiency in tqdm_with_debug(dev_dataloader, debug=args.debug):
                    input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                    labels, efficiency = labels.to(device), efficiency.to(device)
                    logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    loss = loss_func(labels, logits, efficiency)
                    dev_loss.append(loss.item())
                    # result
                    pred = logits.data.cpu().argmax(dim=1)
                    dev_pred_list += pred
                    dev_gold_list += list(np.array(labels.data.cpu()))
                    # score
                    f1 = f1_score(labels.data.cpu(), pred, average='micro')
                    f1_dev.append(f1)
                logger.log('dev_loss:{}'.format(np.mean(dev_loss)))
                logger.log('f1_dev:{}'.format(np.mean(f1_dev)))
                logger.log('dev_ans_confusion_matrix:\n {}'.format(confusion_matrix(dev_gold_list, dev_pred_list)))
                if temp_best < np.mean(f1_dev):
                    temp_best = np.mean(f1_dev)
        temp_best_values.append(temp_best)
        if np.mean(temp_best_values) > max_metric:
            best_epoch = i
            max_metric = np.mean(temp_best_values)
            if args.debug == '1':
                save_model(model=model,
                           model_path='../data/trained_model/',
                           model_name='{}_{}_seed{}_fi{}_gpu{}_be{}_{:.8f}.pth'
                           .format(args.model_name, start_time, args.seed, args.fold_id, args.gpu_type, best_epoch, max_metric),
                           debug=args.debug)
            else:
                save_model(model=model,
                           model_path='../data/trained_model/',
                           model_name='{}.pth'
                           .format(args.model_name),
                           debug=args.debug)

        print('current result: {}'.format(np.mean(temp_best_values), 'best result: {}'.format(max_metric), 'best epoch: {}'.format(best_epoch)))
        logger.log('current result: {}'.format(np.mean(temp_best_values)))
        logger.log('best result: {}'.format(max_metric))
        logger.log('best epoch: {}\n\n\n'.format(best_epoch))


def train_without_dev_binary():
    # log
    start_time = get_time_str()
    model_name = 'pretrained_{}_{}_seed{}_gpu{}.pth'.format(args.model_name, start_time, args.seed, args.gpu_type)
    logger = MyLogger(log_file=log_path + 'time{} {}.log'.format(start_time, model_name), debug=args.debug)
    # data
    batch_size = args.batch_size
    if 'albert-chinese-tiny' in PRETRAINED_PATH:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    label_hub = LabelHub('../data/dataset/{}'.format(args.label_file))
    train_dataset = BinaryDataset('../data/dataset/{}'.format(args.train_file), tokenizer, label_hub)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=binary_collate_fn)
    # model
    device = torch.device('cuda:{}'.format(args.cuda))
    pt_model = AutoModel.from_pretrained(PRETRAINED_PATH, config=config)
    model = PTModelBiClassifier(model=pt_model,
                                model_config=config,
                                dropout_p=0.3).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 40, 60, 100], gamma=0.5)
    loss_func = CrossEntropyWithEfficiency().to(device)
    # train
    max_metric = best_epoch = 0
    logger.log('model:\n{}'.format(model))
    for i in range(args.epoch):
        model.train()
        logger.log('epoch {}'.format(i))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        logger.log('time : {}'.format(get_time_str()))
        if i == 1:
            view_gpu_info(args.cuda)
        train_loss, dev_loss, f1_train, f1_dev = [], [], [], []
        for input_ids, token_type_ids, attention_mask, labels, efficiency in tqdm_with_debug(train_dataloader, debug=args.debug):
            input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
            labels = labels.to(device)
            efficiency = efficiency.to(device)
            logits = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
            optimizer.zero_grad()
            loss = loss_func(labels, logits, efficiency)
            train_loss.append(loss.item())
            # results
            pred = logits.data.cpu().argmax(dim=1)
            f1 = f1_score(labels.data.cpu(), pred, average='micro')
            # socre
            f1_train.append(f1)
            # optim
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('f1: {}'.format(np.mean(f1_train)))
        logger.log('train_loss:{}'.format(np.mean(train_loss)))
        logger.log('f1_train:{}'.format(np.mean(f1_train)))
    if args.debug == '1':
        save_model(model=model,
                   model_path='../data/trained_model/',
                   model_name='pretrained_binary_{}_{}_seed{}_fi{}_gpu{}_be{}_{:.8f}.pth'
                   .format(args.model_name, start_time, args.seed, args.fold_id, args.gpu_type, best_epoch, max_metric),
                   debug=args.debug)
    else:
        save_model(model=model,
                   model_path='../data/trained_model/',
                   model_name='pretrained_binary_{}.pth'
                   .format(args.model_name),
                   debug=args.debug)


if __name__ == '__main__':
    print('----------------------------开始train----------------------------')
    log_path = '../data/Log/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', '-c', default=0, type=int)
    parser.add_argument('--lr', '-l', default=5e-5, type=float)
    parser.add_argument('--epoch', '-e', default=150, type=int)
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--model_name', '-m', default='bert_model', type=str)
    parser.add_argument('--seed', '-s', default=1, type=int)
    parser.add_argument('--gpu_type', '-g', default='2080', choices=['2080', 'xp', '1080', '3090'])
    parser.add_argument('--train_type', '-tt', default='kfold', choices=['muti-task', 'kfold', 'rdrop', 'only_train', 'cm', 'binary', 'only_binary'])
    parser.add_argument('--fold', '-f', default=5, type=int)
    parser.add_argument('--fold_id', '-fi', default=0, type=int, help='选用划分的数据组号，来作为训练数据。k折交叉验证与模型融合的结合。只有在kfold训练模式下使用')
    parser.add_argument('--train_file', '-tf', default='cls_augment2.txt')
    parser.add_argument('--pretrained_model', '-ptm', default='bert', choices=['bert', 'xlnet', 'electra', 'roberta', 'albert', 'gpt2', 'gpt2small', 'macbert'], help='huggingface预训练模型名称')
    parser.add_argument('--debug', '-d', default='0', required=True, choices=['0', '1'])
    parser.add_argument('--label_file', '-lb', default='cls_label2id.json')
    parser.add_argument('--trained_model', '-tm', default='', help='经过劣质样本训练后的模型')
    args = parser.parse_args()
    setup_seed(args.seed)
    # ptmodel
    pretrained_model = args.pretrained_model
    print('----------------------------训练开始时间:{}----------------------------'.format(get_time_str()))
    import time
    start_time = time.time()
    if pretrained_model == 'xlnet':
        PRETRAINED_PATH = XLNET_PRETRAINED_PATH
    elif pretrained_model == 'electra':
        PRETRAINED_PATH = ELECTRA_PRETRAINED_PATH
    elif pretrained_model == 'roberta':
        PRETRAINED_PATH = ROBERTA_PRETRAINED_PATH
    elif pretrained_model == 'albert':
        PRETRAINED_PATH = ALBERT_PRETRAINED_PATH
    elif pretrained_model == 'gpt2':
        PRETRAINED_PATH = GPT2_PRETRAINED_PATH
    elif pretrained_model == 'gpt2small':
        PRETRAINED_PATH = GPT2_SMALL_PRETRAINED_PATH
    elif pretrained_model == 'macbert':
        PRETRAINED_PATH = MACBERT_PRETRAINED_PATH
    if args.train_type == 'kfold':
        k_fold_train()
    elif args.train_type == 'rdrop':
        r_drop_train()
    elif args.train_type == 'only_train':
        train_without_dev()
    elif args.train_type == 'cm':
        k_fold_train_with_cm()
    elif args.train_type == 'binary':
        k_fold_train_binary()
    elif args.train_type == 'only_binary':
        train_without_dev_binary()
    else:
        train()
    print('训练结束时间:', get_time_str())
    print('训练时间开销：{}'.format(get_time_dif(start_time)))
