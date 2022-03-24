"""
@Author: hezf 
@Time: 2021/6/8 11:08 
@desc: 负责RDF图的构建以及查询
"""
from rdflib import Graph, Namespace
from utils import to_sparql
from typing import List, Dict, Tuple
import re


class KnowledgeGraph(object):
    def __init__(self, rdf_file):
        super(KnowledgeGraph, self).__init__()
        self.rdf_file = rdf_file
        self.graph = Graph()
        self.init()

    def init(self):
        # print('parsing file...')
        self.graph.parse(self.rdf_file, format='n3')

    def query(self, q):
        """
        在图谱中查询答案
        :param q: sparql语句
        :return: List[str]
        """
        answers = []
        results = self.graph.query(q).bindings
        for result in results:
            for value in result.values():
                value = value.toPython()
                value = value[value.rfind('/')+1:]
                answers.append(value)
        return answers

    def fetch_ans(self, question: str, ans_type: str, entity: str, main_property: List, operator: str,
                  sub_properties: List[Tuple] = None):
        """
        获取最终答案的函数
        :param question: 问题本身
        :param ans_type: 答案类型【比较句、并列句、属性值】
        :param entity: 实体
        :param main_property: 属性名["档位介绍-流量"或者"XX规则"]，并列句有两个属性名，而其他句式只有一个属性名
        :param sub_properties: 约束属性键值对【[(key, value), (key, value), ...]】。不用字典是因为比较句中key有可能相同
        :param operator: 约束算子【min max other】
        :return:
        """
        # 问句特色：
        # 1 比较句：属性名只要一个，比较句的约束属性名肯定为两个；句式一般为：是否和哪个两种；
        # 2 并列句：有多个属性名，少量情况含有min，max算子，只要将属性名分批答案取出即可
        # 3 属性句：最普通的情况
        try:
            for i in range(len(sub_properties)):
                if sub_properties[i][0] == '流量':
                    value = int(sub_properties[i][1])
                    if 1 < value < 55 and entity != '流量加油包':
                        value *= 1024
                    sub_properties[i] = ('流量', value)
                    # print('修改流量的句子是：', question)
            if ans_type == '比较句':
                if len(sub_properties) == 0:
                    return []
                # 处理当预测的约束属性名个数大于两个时的情况
                if len(sub_properties) > 2:
                    key_count = {}
                    for k, v in sub_properties:
                        if k not in key_count:
                            key_count[k] = 0
                        key_count[k] += 1
                    real_k = None
                    for k, count in key_count.items():
                        if count >= 2:
                            real_k = k
                            break
                    if real_k is not None:
                        for k, v in sub_properties:
                            if k != real_k:
                                sub_properties.remove((k, v))
                q0 = to_sparql(entity=entity, main_property=main_property[0], sub_properties=[sub_properties[0]])
                ans0 = self.query(q0)
                if len(sub_properties) >= 2:
                    q1 = to_sparql(entity=entity, main_property=main_property[0], sub_properties=[sub_properties[1]])
                    ans1 = self.query(q1)
                else:
                    ans1 = None
                if len(ans0) > 0:
                    ans0 = ans0[0]
                else:
                    ans0 = None
                if ans1 is not None and len(ans1) > 0:
                    ans1 = ans1[0]
                else:
                    ans1 = None
                keywords = ['哪', '那个']
                flag = False
                for kw in keywords:
                    if kw in question:
                        flag = True
                        break
                # "哪个"类型的问题
                if flag:
                    if ans1 is None:
                        ans = [sub_properties[0][1]]
                    else:
                        if ans0 == ans1:
                            ans = [sub_properties[0][1], sub_properties[1][1]]
                        else:
                            bigger_keywords = ['多', '贵']
                            smaller_keywords = ['便宜', '优惠', '少', '实惠']
                            big_flag = False
                            for bkw in bigger_keywords:
                                if bkw in question:
                                    big_flag = True
                                    break
                            if big_flag:
                                if int(ans0) > int(ans1):
                                    ans = [sub_properties[0][1]]
                                else:
                                    ans = [sub_properties[1][1]]
                            else:
                                if int(ans0) < int(ans1):
                                    ans = [sub_properties[0][1]]
                                else:
                                    ans = [sub_properties[1][1]]
                # "是否"类型的问题
                else:
                    if ans1 is None:
                        ans = ['no']
                    else:
                        if ans0 == ans1:
                            ans = ['yes']
                        else:
                            ans = ['no']
                        # equal_kw = ['一样', '相同', '等价', '等同', '相等']
                        # eq_q = False
                        # for k in equal_kw:
                        #     if question.find(k) != -1:
                        #         eq_q = True
                        #         break
                        # # 相等问题
                        # if eq_q:
                        #     if ans0 == ans1:
                        #         ans = ['yes']
                        #     else:
                        #         ans = ['no']
                        # # 区别问题
                        # else:
                        #     if ans0 != ans1:
                        #         ans = ['yes']
                        #     else:
                        #         ans = ['no']
            # 并列句
            elif ans_type == '并列句':
                ans = []
                if operator == 'min':
                    if len(sub_properties) == 0:
                        return []
                    # 分两段获取答案
                    key = sub_properties[0][0]
                    query_str = to_sparql(entity=entity, main_property='档位介绍-'+key)
                    temp_ans = self.query(query_str)
                    temp_ans = [int(i) for i in temp_ans]
                    if len(temp_ans) == 0:
                        return []
                    first_step_ans = min(temp_ans)
                    for m_p in main_property:
                        query_str = to_sparql(entity=entity, main_property=m_p, sub_properties=[(key, first_step_ans)])
                        ans += self.query(query_str)
                elif operator == 'max':
                    if len(sub_properties) == 0:
                        return []
                    key = sub_properties[0][0]
                    query_str = to_sparql(entity=entity, main_property='档位介绍-'+key)
                    temp_ans = self.query(query_str)
                    temp_ans = [int(i) for i in temp_ans]
                    if len(temp_ans) == 0:
                        return []
                    first_step_ans = max(temp_ans)
                    for m_p in main_property:
                        query_str = to_sparql(entity=entity, main_property=m_p, sub_properties=[(key, first_step_ans)])
                        ans += self.query(query_str)
                # =!=
                else:
                    for m_p in main_property:
                        query_str = to_sparql(entity=entity, main_property=m_p, sub_properties=sub_properties)
                        ans += self.query(query_str)
            # 属性值
            elif ans_type == '属性值':
                # 答案只有一个，只需要选择一个最小（大）的即可。分两阶段获取答案
                if operator == 'min':
                    if len(sub_properties) == 0:
                        # 直接来近似得到
                        temp_ans = self.query(to_sparql(entity=entity, main_property=main_property[0]))
                        temp_ans = [int(i) for i in temp_ans]
                        ans = min(temp_ans)
                        return [ans]
                    else:
                        key = sub_properties[0][0]
                        query_str = to_sparql(entity=entity, main_property=main_property[0].split('-')[0]+'-'+key)
                        temp_ans = self.query(query_str)
                        temp_ans = [int(i) for i in temp_ans]
                        first_step_ans = min(temp_ans)
                        query_str = to_sparql(entity=entity, main_property=main_property[0], sub_properties=[(key, first_step_ans)])
                        ans = self.query(query_str)
                elif operator == 'max':
                    if len(sub_properties) == 0:
                        query_str = to_sparql(entity=entity, main_property=main_property[0])
                        temp_ans = self.query(query_str)
                        temp_ans = [int(i) for i in temp_ans]
                        ans = max(temp_ans)
                        return [ans]
                    else:
                        key = sub_properties[0][0]
                        query_str = to_sparql(entity=entity, main_property=main_property[0].split('-')[0]+'-'+key)
                        temp_ans = self.query(query_str)
                        # 修复字符串'700'大于'10000'的bug
                        temp_ans = [int(i) for i in temp_ans]
                        first_step_ans = max(temp_ans)
                        query_str = to_sparql(entity=entity, main_property=main_property[0], sub_properties=[(key, first_step_ans)])
                        ans = self.query(query_str)
                else:
                    query_str = to_sparql(entity=entity, main_property=main_property[0], sub_properties=sub_properties)
                    ans = self.query(query_str)
            else:
                ans = []
                print('-------------------{}:乱码对结果造成影响---------------------'.format(ans_type))
            ans = list(set(ans))
            return ans
        except Exception as e:
            print('---------------查找知识图谱时发生异常:{}------------------------'.format(e))
            print('当前参数为，问题：{}, 答案类型：{}, 实体：{}, 属性名{}, 算子：{}, 约束属性：{}'.format(question, ans_type, entity, main_property, operator, sub_properties))
            return []

    def fetch_wrong_ans(self, question: str, ans_type: str, entity: str, main_property: List, operator: str,
                  sub_properties: List[Tuple] = None):
        """
        在没有约束传入时，查看对应的子业务、价格、流量有哪些，从而可以与原句对比，看是否有遗漏的约束标注
        :param question: 问题本身
        :param ans_type: 答案类型【比较句、并列句、属性值】
        :param entity: 实体
        :param main_property: 属性名["档位介绍-流量"或者"XX规则"]，并列句有两个属性名，而其他句式只有一个属性名
        :param sub_properties: 约束属性键值对【[(key, value), (key, value), ...]】。不用字典是因为比较句中key有可能相同
        :param operator: 约束算子【min max other】
        :return:
        """
        # 问句特色：
        # 1 比较句：属性名只要一个，比较句的约束属性名肯定为两个；句式一般为：是否和哪个两种；
        # 2 并列句：有多个属性名，少量情况含有min，max算子，只要将属性名分批答案取出即可
        # 3 属性句：最普通的情况
        for i in range(len(sub_properties)):
            if sub_properties[i][0] == '流量':
                value = int(sub_properties[i][1])
                if 1 < value < 55:
                    value *= 1024
                sub_properties[i] = ('流量', value)
                # print('修改流量的句子是：', question)
        ans, query_str = [], ''
        # 比较句不需要判断，认为约束的标注必然是对的
        if ans_type == '比较句':
            # pass
            # 判断句仍然需要标注。。。
            price_property = '档位介绍-价格'
            flow_property = '档位介绍-流量'
            service_property = '档位介绍-子业务'
            query_str = to_sparql(entity=entity, main_property=price_property, sub_properties=[])
            price_ans = self.query(query_str)
            query_str = to_sparql(entity=entity, main_property=flow_property, sub_properties=[])
            flow_ans = self.query(query_str)
            query_str = to_sparql(entity=entity, main_property=service_property, sub_properties=[])
            service_ans = self.query(query_str)
        # 并列句不需要判断，认为约束一般都是没有的
        elif ans_type == '并列句':
            # pass
            # 并列句仍然需要标注
            price_property = '档位介绍-价格'
            flow_property = '档位介绍-流量'
            service_property = '档位介绍-子业务'
            query_str = to_sparql(entity=entity, main_property=price_property, sub_properties=[])
            price_ans = self.query(query_str)
            query_str = to_sparql(entity=entity, main_property=flow_property, sub_properties=[])
            flow_ans = self.query(query_str)
            query_str = to_sparql(entity=entity, main_property=service_property, sub_properties=[])
            service_ans = self.query(query_str)
        # 属性句
        else:
            # 答案只有一个，只需要选择一个最小（大）的即可。分两阶段获取答案
            if operator == 'min':
                if len(sub_properties) == 0:
                    # 直接来近似得到
                    temp_ans = self.query(to_sparql(entity=entity, main_property=main_property[0]))
                    temp_ans = [int(i) for i in temp_ans]
                    ans = min(temp_ans)
                    return [ans]
                else:
                    key = sub_properties[0][0]
                    query_str = to_sparql(entity=entity, main_property=main_property[0].split('-')[0]+'-'+key)
                    temp_ans = self.query(query_str)
                    temp_ans = [int(i) for i in temp_ans]
                    first_step_ans = min(temp_ans)
                    query_str = to_sparql(entity=entity, main_property=main_property[0], sub_properties=[(key, first_step_ans)])
                    ans = self.query(query_str)
            elif operator == 'max':
                if len(sub_properties) == 0:
                    query_str = to_sparql(entity=entity, main_property=main_property[0])
                    temp_ans = self.query(query_str)
                    temp_ans = [int(i) for i in temp_ans]
                    ans = max(temp_ans)
                    return [ans]
                else:
                    key = sub_properties[0][0]
                    query_str = to_sparql(entity=entity, main_property=main_property[0].split('-')[0]+'-'+key)
                    temp_ans = self.query(query_str)
                    # 修复字符串'700'大于'10000'的bug
                    temp_ans = [int(i) for i in temp_ans]
                    first_step_ans = max(temp_ans)
                    query_str = to_sparql(entity=entity, main_property=main_property[0], sub_properties=[(key, first_step_ans)])
                    ans = self.query(query_str)
            else:
                # if '-' in main_property[0]:
                price_property = '档位介绍-价格'
                flow_property = '档位介绍-流量'
                service_property = '档位介绍-子业务'
                query_str = to_sparql(entity=entity, main_property=price_property, sub_properties=[])
                price_ans = self.query(query_str)
                query_str = to_sparql(entity=entity, main_property=flow_property, sub_properties=[])
                flow_ans = self.query(query_str)
                query_str = to_sparql(entity=entity, main_property=service_property, sub_properties=[])
                service_ans = self.query(query_str)
        price_ans = list(set(price_ans))
        flow_ans = list(set(flow_ans))
        service_ans = list(set(service_ans))
        return (price_ans, flow_ans, service_ans)

if __name__ == '__main__':
    kg = KnowledgeGraph('../data/process_data/triples.rdf')
    # q = 'select ?ans where {<http://yunxiaomi.com/流量加油包> <http://yunxiaomi.com/档位介绍> ?instance. ?instance <http://yunxiaomi.com/流量> ?ans}'
    # ans = kg.query(q)
    ans = kg.fetch_ans(**{'question': '你好！花季守护的软件，需要在孩子的手机里安装东西吗？', 'ans_type': '属性值', 'entity': '花季守护业务', 'main_property': ['档位介绍-使用方法'], 'operator': 'other', 'sub_properties': []})
    print(ans)

    import pandas as pd
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
    df_save.to_excel('/data/huangbo/project/Tianchi_nlp_git/data/raw_data/train_wrong_triple.xlsx', index=False)
