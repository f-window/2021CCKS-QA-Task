"""
@Author: hezf
@Time: 2021/6/19 17:20
@desc: 统计数据模块
"""
import os
import pandas as pd
import re
import json


def statistic_entity(labeled_file: str = '../data/raw_data/train.xlsx'):
    """
    统计”实体“分布
    :param labeled_file:
    :return:
    """
    labeled_data = pd.read_excel(labeled_file)
    entity_list = list(labeled_data.loc[:, '实体'])
    entity_count = {}
    for entity in entity_list:
        if entity not in entity_count:
            entity_count[entity] = 0
        entity_count[entity] += 1
    sorted_dict = sorted(entity_count.items(), key=lambda item: (item[1], item[0]), reverse=True)
    # for entity, count in sorted_dict:
    #     print('实体：{}, 数量：{}'.format(entity, count))
    with open('../data/file/entity_count.json', 'w', encoding='utf-8') as f:
        json.dump(entity_count, f, ensure_ascii=False, indent=2)


def statistic_property(labeled_file: str = '../data/raw_data/train.xlsx'):
    """
    统计”属性名“分布
    :param labeled_file:
    :return:
    """
    labeled_data = pd.read_excel(labeled_file)
    property_list = list(labeled_data.loc[:, '属性名'])
    property_count = {}
    for prop in property_list:
        prop = prop.split('|')
        for p in prop:
            if p not in property_count:
                property_count[p] = 0
            property_count[p] += 1
    sorted_dict = sorted(property_count.items(), key=lambda item: (item[1], item[0]), reverse=True)
    for prop, count in sorted_dict:
        print('属性名：{}, 数量：{}'.format(prop, count))
    with open('../data/file/prop_count.json', 'w', encoding='utf-8') as f:
        json.dump(property_count, f, ensure_ascii=False, indent=2)


def statistic_frequent_char(label_file: str = '../data/dataset/cls_labeled.txt'):
    """
    统计频繁使用的字符，据此推断停用词
    :param label_file:
    :return:
    """
    char_count = {}
    less_5 = set()
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            question = line.split('\t')[0]
            for c in question:
                if c not in char_count:
                    char_count[c] = 0
                char_count[c] += 1
    sorted_dict = sorted(char_count.items(), key=lambda item: (item[1], item[0]), reverse=True)
    for c, count in sorted_dict:
        if count < 10:
            less_5.add(c)
        print('字符：{}, 数量：{}'.format(c, count))
    print('少于5个字符的有：')
    for c in less_5:
        print(c)


def get_all_service(data):
    """
    得到子业务名称
    @param data: excel读取的dataframe数据
    @return: 包含所有子业务名称的list
    """
    service_name = set()
    for index, row in data.iterrows():
        constraint_names = row['约束属性名']
        constraint_values = row['约束属性值']
        constraint_names_list = re.split(r'[｜\|]', str(constraint_names))
        constraint_values_list = re.split(r'[｜\|]', str(constraint_values))
        constraint_names_list = [name.strip() for name in constraint_names_list]
        constraint_values_list = [value.strip() for value in constraint_values_list]
        for i in range(len(constraint_names_list)):
            if constraint_names_list[i] == '子业务':
                service_name.add(constraint_values_list[i])
    return list(service_name)


def statistic_duplicate_constraint(labeled_file: str):
    """
    统计约束属性值的子业务中，值相同的有哪些  //  结果 10条，全是'合约版'
    @param labeled_file:
    @return:
    """
    duplicate = 0

    labeled_data = pd.read_excel(labeled_file)
    con_names = labeled_data['约束属性名'].tolist()
    con_values = labeled_data['约束属性值'].tolist()
    con_names = [re.split(r'[｜\|]', str(item)) for item in con_names]
    con_values = [re.split(r'[｜\|]', str(item)) for item in con_values]
    for index in range(len(con_names)):
        services = []
        con_name = con_names[index]
        con_value = con_values[index]
        for name_id in range(len(con_name)):
            if con_name[name_id] == '子业务':
                if con_value[name_id] in services:
                    print(services)
                    duplicate += 1
                services.append(con_value[name_id])
    print(duplicate)


def statistic_synonyms():
    """
    统计包含的同义词，英文全小写。并将清洗后的同义词写入json文件中
    :return:
    """
    def in_ner_set(word):
        for ner in ner_set:
            if ner in word:
                return True
        return False
    ner_set = {'咪咕直播流量包', '上网版', '成员', '咪咕', '热剧vip', 'plus版', 'pptv', '优酷', '百度', '腾讯视频', '2020版',
                    '大陆及港澳台版', '芒果', '乐享版', '小额版', '年包', '体验版-12个月', '电影vip', '合约版', '优酷会员', '王者荣耀',
                    '长期版', '宝藏版', '免费版', '流量套餐', '乐视会员', '2019版', '基础版', '月包', '全球通版', '畅享包', '腾讯',
                    '爱奇艺会员', '喜马拉雅', '普通版', '流量包', '芒果tv', '百度包', '24月方案', '2018版', '个人版', '半年包',
                    '咪咕流量包', '爱奇艺', '阶梯版', '12月方案', '网易', '家庭版', '198', '5', '12', '160', '20', '220', '398', '15',
                    '200', '24', '30', '3', '288', '18', '238', '120',
                    '128', '60', '9', '10', '300', '188', '59', '68', '8', '38', '2', '80', '680', '70', '158', '1',
                    '380', '298', '11', '65', '40', '19', '23', '29', '6', '99', '500', '22', '49', '100', '40', '700', '110',
                    '20', '100', '500', '1', '30', '300'}
    # 有实体冲突的同义词：（语音留言、家庭亲情号、手机视频流量包）
    conflict = ('语音留言', '家庭亲情号', '手机视频流量包', '流量包')
    synonym_list = []
    synonym2entity = dict()
    for_cls = True
    with open('../data/raw_data/synonyms.txt', 'r', encoding='utf-8') as f:
        for line in f:
            blocks = line.strip().lower().split('	')
            entity, temp_synonym_list = blocks[0], blocks[1].split('|')
            if for_cls or not in_ner_set(entity):
                for s in temp_synonym_list:
                    s = s.strip()
                    if s != '无' and s != '' and s not in conflict:
                        # 为分类模型考虑的同义词增强
                        if for_cls or not in_ner_set(s):
                            synonym2entity[s] = entity
                            synonym_list.append(s)
    synonym_list.sort(key=lambda item: len(item), reverse=True)
    if not for_cls:
        remove_synonym = list()
        for i, synonym in enumerate(synonym_list):
            for j in range(i):
                if synonym in synonym_list[j] and synonym2entity[synonym] != synonym2entity[synonym_list[j]]:
                    # 冲突则移除该词，降低误差
                    print('冲突：', synonym, synonym_list[j])
                    remove_synonym += [synonym, synonym_list[j]]
        # print('需要移除的同义词有：', set(remove_synonym))
        for r_s in set(remove_synonym):
            synonym2entity.pop(r_s)
    entity2synonym = {}
    for s, e in synonym2entity.items():
        if e not in entity2synonym:
            entity2synonym[e] = []
        entity2synonym[e].append(s)
    with open('../data/file/synonyms{}.json'.format('' if for_cls else '_ner'), 'w', encoding='utf-8') as f:
        json.dump({'entity2synonym': entity2synonym, 'synonym2entity': synonym2entity},
                  f,
                  ensure_ascii=False,
                  indent=2)
    print('同义词写入成功...')


def statistic_prop_labels():
    """
    统计属性名的占样本的比例
    [0.241, 0.32, 0.0542, 0.0564, 0.0384, 0.085, 0.0382, 0.129, 0.1276, 0.006, 0.014, 0.006, 0.0014, 0.007, 0.009, 0.0032, 0.0014, 0.001, 0.0004]
    [0.759, 0.68, 0.9458, 0.9436, 0.9616, 0.915, 0.9618, 0.871, 0.8724, 0.994, 0.986, 0.994, 0.9986, 0.993, 0.991, 0.9968, 0.9986, 0.999, 0.9996]

    [0.75, 0.91, 0.87, 0.67, 0.99, 0.94, 0.87, 0.96, 0.99, 0.99, 0.99, 0.96, 0.94, 0.98]
    """
    with open('../data/dataset/cls_label2id_fewer.json', 'r', encoding='utf-8') as f:
        label2id = json.load(f)['main_property']
    count = 0
    prop_count = [0] * len(label2id)
    alpha_count = []
    with open('../data/dataset/cls_labeled_fewer.txt', 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            blocks = line.strip().split('\t')
            props = blocks[2].split('|')
            for p in props:
                if p not in label2id:
                    print(line)
                prop_count[label2id[p]] += 1
    for i in range(len(prop_count)):
        prop_count[i] /= count
        alpha_count.append(1 - prop_count[i])
    print(prop_count)
    print(alpha_count)


def statistic_service(labeled_file: str = '../data/raw_data/train_denoised.xlsx'):
    """

    :param labeled_file:
    :return:
    """
    labeled_data = pd.read_excel(labeled_file)
    name_list = list(labeled_data.loc[:, '约束属性名'])
    value_list = list(labeled_data.loc[:, '约束属性值'])
    service = set()
    price = set()
    flow = set()
    for i, name in enumerate(name_list):
        n_list = str(name).split('|')
        v_list = str(value_list[i]).split('|')
        for j, n in enumerate(n_list):
            if n == '子业务':
                service.add(v_list[j])
            elif n == '流量':
                flow.add(v_list[j])
            elif n == '价格':
                price.add(v_list[j])
    print('所有的子业务:', service)
    print('所有的价格：', price)
    print('所有的流量：', flow)


def statistic_wrong_rdf():
    '''
    向kg输入train.xlsx前面的字段，输出预测结果，并和给出的答案做对比
    @return: train_wrong_triple.xlsx : 保存对比结果
    '''
    print('正在获取rdf结果并对比')
    from triples import KnowledgeGraph

    kg = KnowledgeGraph('../data/process_data/triples.rdf')
    df = pd.read_excel('../data/raw_data/train_denoised.xlsx')
    df.fillna('')
    id_list = []
    ans_list = []
    for iter, row in df.iterrows():
        ans_true = list(set(row['答案'].split('|')))
        question = row['用户问题']
        ans_type = row['答案类型']
        entity = row['实体']
        main_property = row['属性名'].split('|')
        operator = row['约束算子']
        if operator != 'min' and operator != 'max':
            operator == 'other'
        sub_properties = []
        cons_names = str(row['约束属性名']).split('|')
        cons_values = str(row['约束属性值']).split('|')
        if cons_names == ['nan']: cons_names = []
        for index in range(len(cons_names)):
            sub_properties.append([cons_names[index], cons_values[index]])
        ans = kg.fetch_ans(question, ans_type, entity, main_property, operator, sub_properties)

        def is_same(ans, ans_true):
            for an in ans:
                if an in ans_true:
                    ans_true.remove(an)
                else:
                    return False
            if len(ans_true) != 0:
                return False
            return True

        if not is_same(ans, ans_true):
            id_list.append(iter)
            ans_list.append(ans)
    print(id_list)
    df_save = df.iloc[id_list, [0, 1, 2, 3, 4, 5, 6, 7]]
    df_save['预测'] = ans_list
    print(df_save)
    df_save.to_excel('/data/huangbo/project/Tianchi_nlp_git/data/raw_data/train_wrong_triple.xlsx')


def statistic_wrong_cons():
    '''
    找出属性句中（min max不要，并且只要'档位介绍-xx'）中原句中出现但约束中却没出现的价格、子业务、流量
    @return:
    '''
    print('正在获取rdf结果并对比')
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
            if cons_names[index] == '子业务':
                sub_properties[cons_names[index]].append(cons_values[index])
            else:
                sub_properties[cons_names[index]].append(int(cons_values[index]))
        price_ans, flow_ans, service_ans = kg.fetch_wrong_ans(question, ans_type, entity, main_property, operator, [])
        rdf_properties = {}
        rdf_properties['价格'] = price_ans
        rdf_properties['流量'] = flow_ans
        rdf_properties['子业务'] = service_ans
        compare_result = []
        for name, values in rdf_properties.items():
            for value in values:
                if name != '子业务': value = int(value)
                if name == '流量' and (value > 99 and value % 1024 == 0):
                    value = int(value // 1024)
                if str(value) in question:
                    if name in sub_properties:
                        if value in sub_properties[name]:
                            continue
                        if value == '年包' and '半年包' in sub_properties[name]:
                            continue
                        if value == '百度' and '百度包' in sub_properties[name]:
                            continue
                    elif str(value) in entity:
                        continue
                    compare_result.append(name + '_' + str(value))
                    id_list.add(iter)
        if compare_result != []:
            ans_list.append(compare_result)

    index = list(id_list)
    index.sort()
    df_save = df.iloc[index, [0, 1, 2, 3, 4, 5, 6, 7]]
    df_save['预测'] = ans_list
    df_save.to_excel('../data/raw_data/train_wrong_cons.xlsx')


def statistic_wrong_cons_bieo():
    '''
    找出属性句中（min max不要，并且只要'档位介绍-xx'）中原句中出现但约束中却没出现的价格、子业务、流量
    根据bieo的结果进行筛选，补全bieo
    @return:
    '''
    print('正在通过rdf补充信息')
    from triples import KnowledgeGraph
    from data import create_test_BIEO, create_test_BIO

    kg = KnowledgeGraph('../data/process_data/triples.rdf')
    with open('../data/file/train_bieo.json') as f:
        bieo_dict = json.load(f)
    # bieo_dict = create_test_BIEO('../data/raw_data/train_denoised_desyn.xlsx', False)
    type_map = {'价格': 'PRICE', '流量': 'FLOW', '子业务': 'SERVICE'}

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
            pass
            #continue
        entity = row['实体']
        main_property = row['属性名'].split('|')
        # 排除属性中没有'-'的情况，只要'档位介绍-xx'的情况
        if '-' not in main_property[0]:
            # continue
            pass
        operator = row['约束算子']
        # 排除operator为min或max的情况
        if operator != 'min' and operator != 'max':
            operator == 'other'
        else:
            continue
        anno = bieo_dict[question]
        sub_properties = {}
        cons_names = str(row['约束属性名']).split('|')
        cons_values = str(row['约束属性值']).split('|')
        if cons_names == ['nan']: cons_names = []
        for index in range(len(cons_names)):
            if cons_names[index] not in sub_properties:
                sub_properties[cons_names[index]] = []
            if cons_names[index] == '子业务':
                sub_properties[cons_names[index]].append(cons_values[index])
            else:
                sub_properties[cons_names[index]].append(int(cons_values[index]))
        price_ans, flow_ans, service_ans = kg.fetch_wrong_ans(question, ans_type, entity, main_property, operator, [])
        rdf_properties = {}
        rdf_properties['价格'] = price_ans
        rdf_properties['流量'] = flow_ans
        rdf_properties['子业务'] = service_ans
        compare_result = []
        for name, values in rdf_properties.items():
            for value in values:
                if name != '子业务': value = int(value)
                if name == '流量' and (value > 99 and value%1024 == 0):
                    value = int(value//1024)
                if str(value) in question:
                    if name in sub_properties and value in sub_properties[name]:
                        continue
                    if value == '百度' and '百度包' in sub_properties['子业务']:
                        continue
                    question_index = 0
                    if str(value) in entity and (question.count(str(value)) == 1 or str(value) == '1'):
                        continue
                    while question[question_index:].find(str(value)) != -1:
                        temp_index = question_index + question[question_index:].find(str(value))
                        question_index = min(len(question), temp_index + len(str(value)))
                        if anno[temp_index] == 'O':
                            if name == '流量':
                                if question_index < len(question):
                                    if question[question_index] == '元' or question[question_index].isnumeric():
                                        continue
                            if name == '价格':
                                if question_index < len(question):
                                    if question[question_index] == 'g' or question[question_index].isnumeric():
                                        continue
                            if len(str(value)) == 1:
                                anno[temp_index] = 'S-' + type_map[name]
                            else:
                                for temp_i in range(temp_index + 1, temp_index + len(str(value)) - 1):
                                    anno[temp_i] = 'I-' + type_map[name]
                                anno[temp_index] = 'B-' + type_map[name]
                                anno[temp_index + len(str(value)) - 1] = 'E-' + type_map[name]
                            compare_result.append(name + '_' + str(value))
                            bieo_dict[question] = anno
                            id_list.add(iter)

        if compare_result != []:
            ans_list.append(compare_result)

    index = list(id_list)
    index.sort()
    df_save = df.iloc[index, [0, 1, 2, 3, 4, 5, 6, 7]]
    df_save['预测'] = ans_list
    df_save.to_excel('../data/raw_data/train_wrong_cons.xlsx')
    with open(r'../data/file/train_bieo_enhence.json', 'w') as f:
        json.dump(bieo_dict, f, indent=2, ensure_ascii=False)


def statistic_sym_in_question():
    from data_argument import read_synonyms
    synonyms = read_synonyms()
    df = pd.read_excel('../data/raw_data/ensemble_bert_aug2_use_efficiency_2021-07-20-08-29-22_seed1_fi0_gpu2080_be81_0.95411990.pth.xlsx')
    # df = pd.read_excel(
    #     '../data/raw_data/train_denoised.xlsx')
    c = 0
    no = set()
    yes = set()
    for iter, row in df.iterrows():
        entity = row['实体']
        question = row['用户问题']
        con_name = row['约束属性名']
        con_value = row['约束属性值']
        con_names = re.split(r'[｜\|]', str(con_name))
        con_values = re.split(r'[｜\|]', str(con_value))
        if entity not in synonyms: continue
        for synonym in synonyms[entity]:
            if synonym in question:
                for value in con_values:
                    if value in synonym:
                        no.add(synonym)
                if synonym not in no:
                    yes.add(synonym)
    print(no)
    print(yes)
    for iter, row in df.iterrows():
        entity = row['实体']
        question = row['用户问题']
        con_name = row['约束属性名']
        con_value = row['约束属性值']
        con_names = re.split(r'[｜\|]', str(con_name))
        con_values = re.split(r'[｜\|]', str(con_value))
        if entity not in synonyms: continue
        new_question = question
        for synonym in synonyms[entity]:
            if synonym in question and (synonym not in no):
                new_question = question.replace(synonym, entity)
        row['用户问题'] = new_question
    # df.to_excel('../data/raw_data/train_denoised_desyn.xlsx')


    df = pd.read_excel(
        '../data/raw_data/train_denoised.xlsx')
    c = 0
    no = set()
    yes = set()
    for iter, row in df.iterrows():
        entity = row['实体']
        question = row['用户问题']
        con_name = row['约束属性名']
        con_value = row['约束属性值']
        con_names = re.split(r'[｜\|]', str(con_name))
        con_values = re.split(r'[｜\|]', str(con_value))
        if entity not in synonyms: continue
        for synonym in synonyms[entity]:
            if synonym in question:
                for value in con_values:
                    if value in synonym:
                        no.add(synonym)
                if synonym not in no:
                    yes.add(synonym)
    print(no)
    print(yes)
    for iter, row in df.iterrows():
        entity = row['实体']
        question = row['用户问题']
        con_name = row['约束属性名']
        con_value = row['约束属性值']
        con_names = re.split(r'[｜\|]', str(con_name))
        con_values = re.split(r'[｜\|]', str(con_value))
        if entity not in synonyms: continue
        new_question = question
        for synonym in synonyms[entity]:
            if synonym in question and (synonym not in no):
                new_question = question.replace(synonym, entity)
        row['用户问题'] = new_question


# def compare_result(result1_file, result2_file):
#     """
#
#     :param result1_file: 原结果
#     :param result2_file: 新结果
#     :return:
#     """
#     result_path = '../data/results/'
#     result1 = json.load(open(os.path.join(result_path, result1_file), 'r'))
#     result2 = json.load(open(os.path.join(result_path, result2_file), 'r'))
#     for idx, value in result1['model_result'].items():
#         if value != result2['model_result'][idx]:
#             print('结果1：', value, '\n结果2：', result2['model_result'][idx])
#             print('\n')


def string2result(txt):
    with open(txt, 'r') as f:
        line = f.read()
    result_dict = {}
    str_list = line.split('+')
    for str in str_list[:-1]:
        pro_list = str.split('=')
        id = pro_list[0]
        result_dict[id] = {}
        result_dict[id]['question'] = pro_list[1]
        result_dict[id]['ans_type'] = pro_list[2]
        result_dict[id]['entity'] = pro_list[3]
        result_dict[id]['main_property'] = pro_list[4].split('x')
        result_dict[id]['operator'] = pro_list[5]
        result_dict[id]['sub_properties'] = []
        if pro_list[6] != '':
            sub_list = pro_list[6].split('x')
            for sub in sub_list:
                name = sub.split('_')[0]
                value = sub.split('_')[1]
                if name == '流量': value = int(value)
                if pro_list[5] == 'max' or pro_list[5] == 'min':
                    value = int(value)
                result_dict[id]['sub_properties'].append((name, value))

    with open('../data/results/result_docker.json', 'w', encoding='utf-8') as f:
        json.dump({'model_result':result_dict}, f, ensure_ascii=False, indent=2)


def compare_result(file1, file2):
    with open('../data/results/' + file1, 'r') as f:
        base = json.load(f)['model_result']
    with open('../data/results/' + file2, 'r') as f:
        result = json.load(f)['model_result']

    c = 0
    for index, dic in base.items():
        dic1 = result[index]
        # if dic1['sub_properties'] != dic['sub_properties']:
        # if dic1['entity'] != dic['entity']:
        # if dic1['main_property'] != dic['main_property']:
        # if dic1['ans_type'] != dic['ans_type']:
        if dic1['sub_properties'] != dic['sub_properties'] or dic1['entity'] != dic['entity'] or set(dic1['main_property']) != set(dic['main_property']) or dic1['ans_type'] != dic['ans_type']:
            c += 1
            print(index)
            print('file1:', dic)
            print('file2:', dic1)
            print()
    print(c)


def static_confusion_entity():
    with open('../data/dataset/cls_label2id.json', 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    entities = list(label2id['entity'])
    for i in range(len(entities)):
        entity1 = entities[i]
        for j in range(len(entities)):
            entity2 = entities[j]
            if i != j and entity1.find(entity2) != -1:
                print(entity1, 'contains', entity2)


def static_no_answer(file):
    """
    统计答案中没有属性名的样本
    :param file:
    :return:
    """
    with open('../data/results/' + file, 'r') as f:
        result_dict = json.load(f)
    result = result_dict['result']
    model_result = result_dict['model_result']
    for i in result:
        main_property = model_result[i]['main_property']
        for prop in main_property:
            if '-' in prop:
                relation = prop.split('-')[1]
            else:
                relation = prop
            if relation not in ('流量', '价格', '子业务', '上线时间', '语音时长', '带宽', '内含其它服务') and model_result[i]['ans_type'] != '比较句':
                if relation not in result[i]:
                    print('{}\t{}'.format(model_result[i], result[i]))


if __name__ == '__main__':
    # statistic_entity(labeled_file='../data/raw_data/train_denoised.xlsx')
    # statistic_property()
    # statistic_synonyms()
    # statistic_prop_labels()
    # statistic_service()
    # compare_result(result1_file='drop_1_2021-07-28-02-58-01_seed1_gpu2080_be34_0.95407602.pth1.json',
    #                result2_file='drop_1_2021-07-28-02-58-01_seed1_gpu2080_be34_0.95407602.pth.json')
    # statistic_prop_labels()
    # compare_result(result1_file='bert_pt_augment2_2021-07-30-05-44-08_seed1_gpu2080_be6_0.99729231.pth.json',
    #                result2_file='ensemble_bert_aug2_use_efficiency_2021-07-20-08-29-22_seed1_fi0_gpu2080_be81_0.95411990.pth.json')
    string2result('../data/results/result_text.txt')
    compare_result('9677_38.json', '9682.json')
