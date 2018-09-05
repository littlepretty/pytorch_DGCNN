import sys
import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import math
import torch.nn as nn
# import pdb
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.nn.parameter import Parameter
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier
from embedding import EmbedMeanField, EmbedLoopyBP
from util import cmd_args, load_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' %
                os.path.dirname(os.path.realpath(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim +
                             cmd_args.attr_dim,
                             num_edge_feats=0,
                             k=cmd_args.sortpooling_k)
        else:
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim,
                             num_edge_feats=0,
                             max_lv=cmd_args.max_lv)

        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim,
                                 hidden_size=cmd_args.hidden,
                                 num_class=cmd_args.num_class,
                                 with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag is True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag is True:
                tmp = torch.from_numpy(
                    batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag is True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag is True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels)
            # with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag is False and node_tag_flag is True:
            node_feat = node_tag
        elif node_feat_flag is True and node_tag_flag is False:
            pass
        else:
            # use all-one vector as node features
            node_feat = torch.ones(n_nodes, 1)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)

        return self.mlp(embed, labels)

    def embedding(self, graphs):
        node_feat, _ = self.PrepareFeatureLabel(graphs)
        return self.s2v(graphs, node_feat, None)


def fair_sample_prob(indices, labels):
    """
    sample_prob[ith elem] is proportional to 1 / prob[ith elem's label]
    """
    hist, _ = np.histogram(labels, bins=np.arange(max(labels) + 2),
                           density=False)
    dist, _ = np.histogram(labels, bins=np.arange(max(labels) + 2),
                           density=True)
    sample_prob = [1 / x for x in dist]
    sample_prob = [x / sum(sample_prob) for x in sample_prob]  # normalize
    ret = [sample_prob[x] / hist[x] for x in labels]

    return ret


def loop_dataset(g_list, classifier, sample_idxes,
                 optimizer=None, bsize=cmd_args.batch_size):
    total_score = []
    total_iters = (len(sample_idxes) +
                   (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    # graph_labels = [g.label for g in g_list]
    # fair_prob = fair_sample_prob(sample_idxes, graph_labels)

    n_samples = 0
    all_pred = []
    all_label = []
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]
        # selected_idx = np.random.choice(sample_idxes, bsize, fair_prob)
        batch_graph = [g_list[idx] for idx in selected_idx]
        _, loss, acc, precision, recall, pred = classifier(batch_graph)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_pred.extend(pred.data.cpu().numpy().tolist())
        all_label.extend([g.label for g in batch_graph])
        avg_precision = np.nanmean(precision.data.cpu().numpy())
        avg_recall = np.nanmean(recall.data.cpu().numpy())
        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
        total_score.append(np.array([loss, acc, avg_precision, avg_recall]))
        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)

    total_score = np.array(total_score)
    avg_score = np.mean(np.array(total_score), 0)
    return avg_score, all_pred, all_label


def compute_pr_scores(pred, labels, prefix):
    scores = {}
    scores['precisions'] = precision_score(labels, pred, average=None)
    scores['recalls'] = recall_score(labels, pred, average=None)
    df = pd.DataFrame.from_dict(scores)
    df.to_csv('%s_%s_pr_scores.txt' % (cmd_args.data, prefix),
              float_format='%.4f')


def compute_confusion_matrix(pred, labels, prefix):
    cm = confusion_matrix(labels, pred)
    np.savetxt('%s_%s_confusion_matrix.txt' % (cmd_args.data, prefix), cm,
               fmt='%4d', delimiter=' ')


def store_embedding(classifier, graphs, prefix, sample_size=100):
    if len(graphs) > sample_size:
        sample_idx = np.random.randint(0, len(graphs), sample_size)
        graphs = [graphs[i] for i in sample_idx]

    emb = classifier.embedding(graphs)
    emb = emb.data.cpu().numpy()
    labels = [g.label for g in graphs]
    np.savetxt('%s_%s_embedding.txt' % (cmd_args.data, prefix),
               emb, fmt='%8.8f')
    np.savetxt('%s_%s_embedding_label.txt' % (cmd_args.data, prefix),
               labels, fmt='%d')


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs +
                                 test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[
            int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
        ]
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
        # classifier = nn.DataParallel(classifier)

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    train_loss_hist = []
    train_accu_hist = []
    test_loss_hist = []
    test_accu_hist = []
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_score, train_pred, train_labels = \
            loop_dataset(train_graphs, classifier,
                         train_idxes, optimizer=optimizer)
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f precision %.5f recall %.5f\033[0m' %
              (epoch, avg_score[0], avg_score[1], avg_score[2], avg_score[3]))
        train_loss_hist.append(avg_score[0])
        train_accu_hist.append(avg_score[1])

        classifier.eval()
        test_score, test_pred, test_labels = \
            loop_dataset(test_graphs, classifier,
                         list(range(len(test_graphs))))
        print('\033[93maverage testing of epoch %d:  loss %.5f acc %.5f precision %.5f recall %.5f\033[0m' %
              (epoch, test_score[0], test_score[1],
               test_score[2], test_score[3]))
        test_loss_hist.append(test_score[0])
        test_accu_hist.append(test_score[1])
        if epoch + 1 == cmd_args.num_epochs:
            compute_pr_scores(test_pred, test_labels, 'test')
            compute_confusion_matrix(train_pred, train_labels, 'train')
            compute_confusion_matrix(test_pred, test_labels, 'test')
            store_embedding(classifier, train_graphs, 'train')
            store_embedding(classifier, test_graphs, 'test')

    hist = {}
    hist['train_loss'] = train_loss_hist
    hist['train_accu'] = train_accu_hist
    hist['test_loss'] = test_loss_hist
    hist['test_accu'] = test_accu_hist
    df = pd.DataFrame.from_dict(hist)
    df.to_csv('%s_hist.txt' % cmd_args.data, float_format='%.6f')
