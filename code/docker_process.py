import os
import shutil

from data_argument import *
from statistic import *
from utils import *

def prepare_dir():
    '''
    用于生成存放数据的文件夹
    @return:
    '''
    if not os.path.exists('/data'):
        os.mkdir('/data')
    if not os.path.exists('/data/raw_data'):
        os.mkdir('/data/raw_data')
    if not os.path.exists('/data/process_data'):
        os.mkdir('/data/process_data')
    if not os.path.exists('/data/file'):
        os.mkdir('/data/file')
    if not os.path.exists('/data/dataset'):
        os.mkdir('/data/dataset')
    if not os.path.exists('/data/Log'):
        os.mkdir('/data/Log')
    if not os.path.exists('/data/trained_model'):
        os.mkdir('/data/trained_model')
    if not os.path.exists('/data/results'):
        os.mkdir('/data/results')


def prepare_copy_file():
    '''
    为统一路径，将天池数据/tcdata中的内容复制到/data/raw_data中
    @return:
    '''
    if os.path.exists('/tcdata'):
        for file_name in os.listdir('/tcdata'):
            shutil.copy(os.path.join('/tcdata', file_name), '/data/raw_data')

    # # 查看是否复制成功
    # raw_data = os.listdir('/data/raw_data')
    # for file_name in os.listdir('/tcdata'):
    #     if file_name not in raw_data:
    #         print('复制失败: ', file_name)


def prepare_data():
    '''
    一系列文件生成工作
    @return:
    '''
    # 生成denoiseed.xlsx
    denoising(source_file='../data/raw_data/train.xlsx',
              target_file='../data/raw_data/train_denoised.xlsx')
    denoising(source_file='../tcdata/test2.xlsx',
              target_file='../data/raw_data/test_denoised.xlsx')
    # 生成统计文件：实体个数、同义词个数
    statistic_entity()
    statistic_synonyms()
    # entity_map()
    # 生成 数据增强 文件
    augment_for_few_data(source_file='../data/raw_data/train_denoised.xlsx',
                         target_file='../data/raw_data/train_augment_few_nlpcda.xlsx')
    multi_process_augment_from_simbert()
    augment_for_synonyms(source_file='../data/raw_data/train_denoised.xlsx',
                         target_file='../data/raw_data/train_augment_synonyms.xlsx')
    # 生成标注好的训练数据
    make_dataset(['../data/raw_data/train_denoised.xlsx'],
                 target_file='../data/dataset/cls_labeled.txt',
                 label_file='../data/dataset/cls_label2id.json',
                 train=True)
    # 将增强文件合并
    make_dataset(['../data/raw_data/train_augment_few_nlpcda.xlsx',
                  '../data/raw_data/train_augment_simbert.xlsx',
                  '../data/raw_data/train_augment_synonyms.xlsx'],
                 target_file='../data/dataset/augment3.txt',  # 修改此处应该修改run.sh中的文件
                 label_file=None,
                 train=True)
    make_dataset(['../data/raw_data/test_denoised.xlsx'],
                 target_file='../data/dataset/cls_unlabeled.txt',
                 label_file=None,
                 train=False)
    # ner数据增强
    augment_for_ner(source_file='../data/raw_data/train_denoised.xlsx',
                    target_file='../data/raw_data/train_denoised_ner.xlsx')
    # 制作二分类训练数据
    augment_for_binary(source_file='../data/raw_data/train_denoised.xlsx',
                       target_file='../data/raw_data/train_augment_binary.xlsx')
    make_dataset_for_binary(['../data/raw_data/train_denoised.xlsx'],
                            target_file='../data/dataset/binary_labeled.txt')
    make_dataset_for_binary(['../data/raw_data/train_augment_binary.xlsx'],
                            target_file='../data/dataset/binary_augment3.txt')
    # 制作rdf文件
    parse_triples_file('../data/raw_data/triples.txt')


if __name__ == '__main__':
    print('docker运行开始，时间为', get_time_str())
    print('---------------------开始process-----------------------')
    setup_seed(1) # todo
    prepare_dir()
    prepare_copy_file()
    prepare_data()
    print('---------------------process结束-----------------------')
    end = time.time()

