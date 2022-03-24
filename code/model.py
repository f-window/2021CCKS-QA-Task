"""
@Author: hezf 
@Time: 2021/6/3 14:42 
@desc: 
"""
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.optim as optim
from utils import *
from copy import deepcopy
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from itertools import zip_longest
from collections import Counter
from module import Attention, FGM


class PTModel(nn.Module):
    """
    BERT预训练模型进行多任务【多分类、多标签】
    """
    def __init__(self, model, ans_class, prop_label, entity_class, dropout_p=0.1):
        super(PTModel, self).__init__()
        self.model = model
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=dropout_p)
        self.ans_classifier = nn.Linear(768, ans_class)
        self.prop_classifier = nn.Linear(768, prop_label)
        self.entity_classifier = nn.Linear(768, entity_class)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = cls_emb[1]
        cls_emb = self.layer_norm(cls_emb)
        cls_emb = self.dropout(cls_emb)
        ans_logits = self.ans_classifier(cls_emb)
        prop_logits = self.prop_classifier(cls_emb)
        prop_logits = torch.sigmoid(prop_logits)
        entity_logits = self.entity_classifier(cls_emb)
        return ans_logits, prop_logits, entity_logits


class PTModelTaskAttention(nn.Module):
    """
    [设为默认baseline]
    BERT预训练模型进行多任务【多分类、多标签】
    """
    def __init__(self, model, ans_class, prop_label, entity_class, model_config, dropout_p=0.1):
        super(PTModelTaskAttention, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(model_config.hidden_size)
        self.ans_classifier = nn.Linear(model_config.hidden_size, ans_class)
        self.prop_classifier = nn.Linear(model_config.hidden_size, prop_label)
        self.entity_classifier = nn.Linear(model_config.hidden_size, entity_class)
        self.ans_attention = Attention(hidden_size=model_config.hidden_size)
        self.prop_attention = Attention(hidden_size=model_config.hidden_size)
        self.entity_attention = Attention(hidden_size=model_config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        mask = input_ids == 0
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = cls_emb[0]
        cls_emb = self.layer_norm(cls_emb)
        cls_emb = self.dropout(cls_emb)
        ans_logits = self.ans_classifier(self.ans_attention(cls_emb, mask))
        prop_logits = self.prop_classifier(self.prop_attention(cls_emb, mask))
        prop_logits = torch.sigmoid(prop_logits)
        entity_logits = self.entity_classifier(self.entity_attention(cls_emb, mask))
        return ans_logits, prop_logits, entity_logits


class PTModelTaskAttentionV1(nn.Module):
    """
    在baseline的基础上添加了【开通方式、开通条件的判断】
    """
    def __init__(self, model, ans_class, prop_label, entity_class, model_config, dropout_p=0.1):
        super(PTModelTaskAttentionV1, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(model_config.hidden_size)
        self.ans_classifier = nn.Linear(model_config.hidden_size, ans_class)
        self.prop_classifier = nn.Linear(model_config.hidden_size, prop_label)
        self.entity_classifier = nn.Linear(model_config.hidden_size, entity_class)
        # 二分类
        self.method_classifier = nn.Linear(model_config.hidden_size, 2)
        self.condition_classifier = nn.Linear(model_config.hidden_size, 2)

        self.ans_attention = Attention(hidden_size=model_config.hidden_size)
        self.prop_attention = Attention(hidden_size=model_config.hidden_size)
        self.entity_attention = Attention(hidden_size=model_config.hidden_size)
        self.method_attention = Attention(hidden_size=model_config.hidden_size)
        self.condition_attention = Attention(hidden_size=model_config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        mask = input_ids == 0
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = cls_emb[0]
        cls_emb = self.layer_norm(cls_emb)
        cls_emb = self.dropout(cls_emb)
        ans_logits = self.ans_classifier(self.ans_attention(cls_emb, mask))
        prop_logits = self.prop_classifier(self.prop_attention(cls_emb, mask))
        prop_logits = torch.sigmoid(prop_logits)
        entity_logits = self.entity_classifier(self.entity_attention(cls_emb, mask))

        method_logits = self.method_classifier(self.method_attention(cls_emb, mask))
        condition_logits = self.condition_classifier(self.condition_attention(cls_emb, mask))
        return ans_logits, prop_logits, entity_logits, method_logits, condition_logits


class PTModelBiClassifier(nn.Module):
    """
    这里用来做"开通方式和开通条件的判断"
    """
    def __init__(self, model, model_config, dropout_p=0.1):
        super(PTModelBiClassifier, self).__init__()
        self.model = model
        self.attention = Attention(hidden_size=model_config.hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(model_config.hidden_size)
        self.classifier = nn.Linear(model_config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if len(cls_emb) > 1:
            cls_emb = cls_emb[1]
        else:
            cls_emb = cls_emb[0]
            mask = input_ids == 0
            cls_emb = self.attention(cls_emb, mask)
        cls_emb = self.layer_norm(cls_emb)
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)
        return logits


class PTModelAttention(nn.Module):
    """
    BERT预训练模型进行多任务【多分类、多标签】
    """
    def __init__(self, model, ans_class, prop_label, entity_class, dropout_p=0.1):
        super(PTModelAttention, self).__init__()
        self.ans_class = ans_class
        self.prop_label = prop_label
        self.entity_class = entity_class
        self.model = model
        self.dropout = nn.Dropout(p=dropout_p)
        self.attention_ans = nn.Linear(768, 16)
        self.attention_prop = nn.Linear(768, 16)
        self.attention_entity = nn.Linear(768, 16)
        self.ans_classifier = nn.Linear(768, 16*ans_class)
        self.prop_classifier = nn.Linear(768, 16*prop_label)
        self.entity_classifier = nn.Linear(768, 16*entity_class)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = cls_emb[1]
        cls_emb = self.dropout(cls_emb)
        # ans
        attention_value1 = torch.softmax(self.attention_ans(cls_emb), dim=-1).unsqueeze(dim=1)
        ans_logits = self.ans_classifier(cls_emb).view(-1, 16, self.ans_class).contiguous()
        ans_logits = torch.bmm(attention_value1, ans_logits, out=None).squeeze(1)
        # prop
        attention_value2 = torch.softmax(self.attention_prop(cls_emb), dim=-1).unsqueeze(dim=1)
        prop_logits = self.prop_classifier(cls_emb).view(-1, 16, self.prop_label).contiguous()
        prop_logits = torch.bmm(attention_value2, prop_logits, out=None).squeeze(1)
        prop_logits = torch.sigmoid(prop_logits)
        # entity
        attention_value3 = torch.softmax(self.attention_entity(cls_emb), dim=-1).unsqueeze(dim=1)
        entity_logits = self.entity_classifier(cls_emb).view(-1, 16, self.entity_class).contiguous()
        entity_logits = torch.bmm(attention_value3, entity_logits, out=None).squeeze(1)
        return ans_logits, prop_logits, entity_logits


class PTModelForProperty(nn.Module):
    """
    BERT预训练模型单独进行多标签任务
    """
    def __init__(self, model, prop_label, dropout_p=0.1):
        super(PTModelForProperty, self).__init__()
        self.prop_label = prop_label
        self.model = model
        self.dropout = nn.Dropout(p=dropout_p)
        self.attention = nn.Linear(768, 16)
        self.prop_classifier = nn.Linear(768, prop_label*16)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = cls_emb[1]
        cls_emb = self.dropout(cls_emb)

        attention_value = torch.softmax(self.attention(cls_emb), dim=-1).unsqueeze(dim=1)
        prop_logits = self.prop_classifier(cls_emb).view(-1, 16, self.prop_label).contiguous()

        prop_logits = torch.bmm(attention_value, prop_logits, out=None).squeeze(1)
        prop_logits = torch.sigmoid(prop_logits)
        return prop_logits


class PTModelForPropertyV2(nn.Module):
    """
    BERT预训练模型单独进行多标签任务【用多个二分类来完成多标签任务】
    """
    def __init__(self, model, prop_label, dropout_p=0.1):
        super(PTModelForPropertyV2, self).__init__()
        self.prop_label = prop_label
        self.model = model
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier_list = nn.ModuleList([nn.Linear(768, 2) for _ in range(prop_label)])
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = cls_emb[1]
        cls_emb = self.dropout(cls_emb)
        # 添加layernorm
        cls_emb = self.layer_norm(cls_emb)
        logits = []
        for classifier in self.classifier_list:
            logit = classifier(cls_emb)
            logits.append(logit)
        logits = torch.stack(logits).view(-1, 2).contiguous()
        return logits


class PTModelForMultiClassification(nn.Module):
    """
        BERT预训练模型进行多分类任务
    """
    def __init__(self, model, n_class, dropout_p=0.1):
        super(PTModelForMultiClassification, self).__init__()
        self.n_class = n_class
        self.model = model
        self.attention = nn.Linear(768, 16)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(768, 16*n_class)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = cls_emb[1]
        cls_emb = self.dropout(cls_emb)
        attention_value = torch.softmax(self.attention(cls_emb), dim=-1).unsqueeze(dim=1)
        logits = self.classifier(cls_emb).view(-1, 16, self.n_class).contiguous()
        logits = torch.bmm(attention_value, logits, out=None).squeeze(1)
        return logits


class BILSTM_CRF_Model(object):
    def __init__(self, vocab_size, out_size, gpu_id=0, crf=True):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        self.device = torch.device(
            "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size

        self.crf = crf

        self.model = BiLSTM_CRF(vocab_size, self.emb_size,
                                self.hidden_size, out_size).to(self.device)
        self.cal_loss_func = cal_lstm_crf_loss

        # 加载训练参数：
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 初始化学习率优化器
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 30, 40], gamma=0.5)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None
        self.fgm = FGM(self.model)

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id, debug='0'):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        B = self.batch_size
        for e in range(1, self.epoches + 1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind + B]
                batch_tags = tag_lists[ind:ind + B]

                losses += self.train_step(batch_sents,
                                          batch_tags, word2id, tag2id)

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    if debug == '1':
                        print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                            e, self.step, total_step,
                            100. * self.step / total_step,
                            losses / self.print_step
                        ))
                    losses = 0.
            self.scheduler.step()
            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id, debug)
            if debug == '1':
                print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()

        # TODO 对抗攻击还没在测试集验证效果
        # self.fgm.attack('embedding')
        # scores_atk = self.model(tensorized_sents, lengths)
        # loss_atk = self.cal_loss_func(scores_atk, targets, tag2id).to(self.device)
        # loss_atk.backward()
        # self.fgm.restore()

        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id, debug):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind + self.batch_size]
                batch_tags = dev_tag_lists[ind:ind + self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                if debug == '1':
                    print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists

    def test_k_pre(self, word_lists, tag_lists, word2id):
        """将test分成两部分，pre得到得分，post根据得分得到编码"""
        # 准备数据
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            score = self.best_model.test_k_pre(
                tensorized_sents, lengths)

        return score

    def test_k_post(self, word_lists, tag_lists, word2id, tag2id, score):
        """将test分成两部分，pre得到得分，post根据得分得到编码，score应该是5折的平均或最大值结果"""
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)

        batch_tagids = self.best_model.test_k_post(score, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(2*hidden_size, out_size)

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)
        # self.transition.data.zero_()

    def forward(self, sents_tensor, lengths):
        # 添加，以防止警告提醒
        self.bilstm.flatten_parameters()
        # [B, L, out_size]
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        emission = self.lin(rnn_out)  # [B, L, out_size]

        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids

    def test_k_pre(self, test_sents_tensor, lengths):
        crf_scores = self.forward(test_sents_tensor, lengths)

        return crf_scores

    def test_k_post(self, crf_scores, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step - 1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],  # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L - 1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L - 1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids


class BERT_BILSTM_CRF_Model(object):
    def __init__(self, vocab_size, out_size, bert_model, gpu_id=0, crf=True):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        self.device = torch.device(
            "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        self.emb_size = BERTLSTMConfig.emb_size
        self.hidden_size = BERTLSTMConfig.hidden_size

        self.crf = crf

        self.model = BERT_BiLSTM_CRF(vocab_size, self.emb_size,
                                self.hidden_size, out_size, bert_model).to(self.device)
        self.bert_model = bert_model
        self.cal_loss_func = cal_lstm_crf_loss

        # 加载训练参数：
        self.epoches = BertTrainingConfig.epoches
        self.print_step = BertTrainingConfig.print_step
        self.lr = BertTrainingConfig.lr
        self.other_lr = BertTrainingConfig.other_lr
        self.batch_size = BertTrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = self._build_optimizer()  # optim.Adam(self.model.parameters(), lr=self.lr)
        # 初始化学习率优化器
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 30], gamma=0.5)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def _build_optimizer(self):
        module = (self.model.module if hasattr(self.model, "module") else self.model)
        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())
        bert_param_optimizer = []
        other_param_optimizer = []
        for name, para in model_param:
            space = name.split('.')
            if space[0] == 'bert_model':
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.other_lr},
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)
        return optimizer

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        B = self.batch_size
        for e in range(1, self.epoches + 1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind + B]
                batch_tags = tag_lists[ind:ind + B]

                losses += self.train_step(batch_sents,
                                          batch_tags, word2id, tag2id)

                if self.step % BertTrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    # print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                    #     e, self.step, total_step,
                    #     100. * self.step / total_step,
                    #     losses / self.print_step
                    # ))
                    losses = 0.
            self.scheduler.step()
            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized_bert(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind + self.batch_size]
                batch_tags = dev_tag_lists[ind:ind + self.batch_size]
                tensorized_sents, lengths = tensorized_bert(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized_bert(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, bert_model):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BERT_BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(emb_size, out_size)
        self.bert_model = bert_model

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)
        # self.transition.data.zero_()

    def forward(self, sents_tensor, lengths):
        # 添加，以防止警告提醒
        self.bilstm.flatten_parameters()
        # [B, L, out_size]
        # emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        input_ids = sents_tensor
        device = input_ids.device
        token_type_ids = torch.zeros(input_ids.shape).to(torch.int64).to(device)
        one = torch.ones(input_ids.shape).to(torch.int64).to(device)
        attention_mask = torch.where(input_ids>0, one, token_type_ids)
        emb = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        # packed = pack_padded_sequence(emb, lengths, batch_first=True)
        # rnn_out, _ = self.bilstm(packed)
        # # rnn_out:[B, L, hidden_size*2]
        # rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        emission = self.lin(emb)  # [B, L, out_size]

        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids


class Metrics(object):
    """用于评价BiLSTM模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = self.flatten_lists(golden_tags)
        self.predict_tags = self.flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

    def flatten_lists(self, lists):
        flatten_list = []
        for l in lists:
            if type(l) == list:
                flatten_list += l
            else:
                flatten_list.append(l)
        return flatten_list

    def cal_precision(self):

        precision_scores = {}
        for tag in self.tagset:
            try:
                precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.predict_tags_counter[tag]
            except:
                precision_scores[tag] = 0

        return precision_scores

    def cal_recall(self):

        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.golden_tags_counter[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2*p*r / (p+r+1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # 计算并打印平均值
        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))

    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def _remove_Otags(self):

        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size+1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))