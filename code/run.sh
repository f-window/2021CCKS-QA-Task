#!/bin/sh
mode='multi'
debug_status='no'

if test $debug_status = 'yes'
then
  batch_size1=16
  batch_size2=16
  batch_size3=16
  ptm_epoch=1
  ptm_epoch_binary=1
  train_epoch=1
  train_epoch_binary=1
  pt_file=augment3.txt
elif test $debug_status = 'no'
then
  batch_size1=350
  batch_size2=256
  batch_size3=190
  ptm_epoch=14
  ptm_epoch_binary=5
  train_epoch=5
  train_epoch_binary=5
  pt_file=augment3.txt
fi
# TODO 注意回车标志只能是LF
# 单模型
if test $mode = 'single'
then
  python docker_process.py
  # BERT等模型的batch_size=350,XLNet的batch_size=256
  # GPT2batch_size=180暂定,学习率1e-4
  # bert+rdrop的batch_size=180
  python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm roberta -d 0 -m roberta -b $batch_size1 -l 0.0002
  python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm roberta -d 0 -m roberta_model -b $batch_size1 -l 0.00005 -tm pretrained_roberta.pth
  python train_evaluate_constraint.py -m ner_model.pth -d 0 -tv bieo -k 1
  # 修改模型时，这里模型名也要修改
  python inference.py -cm roberta_model.pth -nm ner_model.pth -tv bieo -k 1
# 多模型
elif test $mode = 'multi'
then
  python docker_process.py
  # bert
  python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm bert -d 0 -m bert -b $batch_size1 -l 0.0002
  python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm bert -d 0 -m bert_model -b $batch_size1 -l 0.00005 -tm pretrained_bert.pth
  # roberta
  python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm roberta -d 0 -m roberta -b $batch_size1 -l 0.0002
  python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm roberta -d 0 -m roberta_model -b $batch_size1 -l 0.00005 -tm pretrained_roberta.pth
  # electra
  python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm electra -d 0 -m electra -b $batch_size1 -l 0.0002
  python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm electra -d 0 -m electra_model -b $batch_size1 -l 0.00005 -tm pretrained_electra.pth
  # xlnet
  python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm xlnet -d 0 -m xlnet -b $batch_size2 -l 0.0002
  python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm xlnet -d 0 -m xlnet_model -b $batch_size2 -l 0.00005 -tm pretrained_xlnet.pth
  # gpt2
  #python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm gpt2 -d 0 -m gpt2 -b $batch_size3 -l 0.0001
  #python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm gpt2 -d 0 -m gpt2_model -b $batch_size3 -l 0.00005 -tm pretrained_gpt2.pth
  # macbert
  python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm macbert -d 0 -m macbert -b $batch_size1 -l 0.0002
  python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm macbert -d 0 -m macbert_model -b $batch_size1 -l 0.00005 -tm pretrained_macbert.pth
  # albert
  #python train.py -tt only_train -e $ptm_epoch -tf $pt_file -ptm albert -d 0 -m albert -b $batch_size1 -l 0.001
  #python train.py -tt kfold -e $train_epoch -tf cls_labeled.txt -ptm albert -d 0 -m albert_model -b $batch_size1 -l 0.00025 -tm pretrained_albert.pth
  # 二分类模型
  python train.py -tt only_binary -e $ptm_epoch_binary -tf binary_augment3.txt -ptm bert -d 0 -m bert -b 32 -l 0.00005
  python train.py -tt binary -e $train_epoch_binary -tf binary_labeled.txt -ptm bert -d 0 -m bert_binary -b 32 -l 0.00001 -tm pretrained_binary_bert.pth
  # ner
  python train_evaluate_constraint.py -m ner_model.pth -d 0 -tv bieo -k 5
  # inference
  python inference.py -cm bert_model.pth -nm ner_model.pth -tv bieo -e 1 -de 0 -b 4 -xm xlnet_model.pth -em electra_model.pth -rm roberta_model.pth -mm macbert_model.pth -k 5 -wb
elif test $mode = 'orig'
then
  python docker_process.py
  # BERT等模型的batch_size=350,XLNet的batch_size=256
  python train.py -tt kfold -e 50 -tf cls_labeled.txt -ptm bert -d 0 -m bert_model -b 350 -l 0.0002
  # eh为标注增强，可设为0
  python train_evaluate_constraint.py -m ner_model.pth -d 0 -tv bieo -k 5
  # 修改模型时，这里模型名也要修改, eh为标注增强，可设为0
  python inference.py -cm bert_model.pth -nm ner_model.pth -tv bieo -dp ../data/dataset/cls_unlabeled.txt -k 5
else
  echo "其他"
fi
