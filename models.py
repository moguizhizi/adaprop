import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from   torch_scatter import scatter
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from   collections import defaultdict
import copy

def getSameRatio(tensor1, tensor2):
    combined        = torch.cat((tensor1, tensor2))
    uniques, counts = combined.unique(return_counts=True)
    same_node_idx   = counts > 1  # (True / False)
    return torch.sum(same_node_idx) / tensor1.shape[0]

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_ent, \
                    MESS_FUNC='TransE', AGG_FUNC='max', COMB_FUNC='discard', \
                    n_node_topk=-1, n_edge_topk=-1, tau=1.0, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel       = n_rel
        self.n_ent       = n_ent
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.attn_dim    = attn_dim
        self.act         = act
        self.MESS_FUNC   = MESS_FUNC
        self.AGG_FUNC    = AGG_FUNC
        self.COMB_FUNC   = COMB_FUNC
        self.n_node_topk = n_node_topk
        self.n_edge_topk = n_edge_topk
        self.tau         = tau
        self.rela_embed  = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn    = nn.Linear(in_dim, attn_dim)
        self.w_alpha     = nn.Linear(attn_dim, 1)
        self.W_h         = nn.Linear(in_dim, out_dim, bias=False)
        if self.n_node_topk > 0: self.W_samp = nn.Linear(in_dim, 1, bias=False)
        
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.training and self.tau > 0: 
            self.softmax = lambda x : F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x : F.softmax(x, dim=1)
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize):
        # edges: [N_edge_of_all_batch, 6] 
        # with (batch_idx, head, rela, tail, head_idx, tail_idx)
        # note that head_idx and tail_idx are relative index
        sub     = edges[:,4]
        rel     = edges[:,2]
        obj     = edges[:,5]
        hs      = hidden[sub]
        hr      = self.rela_embed(rel)
        r_idx   = edges[:,0]
        h_qr    = self.rela_embed(q_rel)[r_idx]
        n_node  = nodes.shape[0]

        # MESS()
        if self.MESS_FUNC == 'TransE':   
            message = hs + hr
        elif self.MESS_FUNC == 'DistMult':   
            message = hs * hr
        elif self.MESS_FUNC == 'RotatE':  
            hs_re, hs_im = hs.chunk(2, dim=-1)
            hr_re, hr_im = hr.chunk(2, dim=-1)
            message_re = hs_re * hr_re - hs_im * hr_im
            message_im = hs_re * hr_im + hs_im * hr_re
            message = torch.cat([message_re, message_im], dim=-1)

        # sample edges w.r.t. alpha
        if self.n_edge_topk > 0:
            alpha          = self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))).squeeze(-1)
            edge_prob      = F.gumbel_softmax(alpha, tau=1, hard=False)
            topk_index     = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
            edge_prob_hard = torch.zeros((alpha.shape[0])).cuda()
            edge_prob_hard[topk_index] = 1
            alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
            alpha = torch.sigmoid(alpha).unsqueeze(-1)
        else:
            alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))) # [N_edge_of_all_batch, 1]

        # aggregate message and then propagate
        message = alpha * message

        # AGG()
        if self.AGG_FUNC == 'sum':
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        elif self.AGG_FUNC == 'mean':
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='mean')
        elif self.AGG_FUNC == 'max':
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='max')
        
        # COMB()
        if self.COMB_FUNC == 'discard':
            hidden_new = self.act(self.W_h(message_agg)) # [n_node, dim]
        elif self.COMB_FUNC == 'residual':
            message_agg[old_nodes_new_idx] += hidden
            hidden_new = self.act(self.W_h(message_agg)) # [n_node, dim]

        # forward without node sampling
        if self.n_node_topk <= 0:
            return hidden_new

        # forward with node sampling
        # ============ node sampling starts ============
        # indexing sampling operation
        tmp_diff_node_idx = torch.ones(n_node)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]

        # project logit to fixed-size tensor via indexing
        diff_node_logit  = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1) # [all_batch_new_nodes]

        # save logit to node_scores for later indexing
        node_scores = torch.ones((batchsize, self.n_ent)).cuda() * float('-inf')
        node_scores[diff_node[:,0], diff_node[:,1]] = diff_node_logit

        # select top-k nodes
        # (train mode) self.softmax == F.gumbel_softmax
        # (eval mode)  self.softmax == F.softmax 
        node_scores = self.softmax(node_scores) # [batchsize, n_ent]
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batchidx  = torch.arange(batchsize).repeat(self.n_node_topk,1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batchsize, self.n_ent)).cuda()
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        # get sampled nodes' relative index
        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:,0], diff_node[:,1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx.cuda()
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx

        # update node embeddings
        diff_node_prob_hard = batch_topk_nodes[diff_node[:,0], diff_node[:,1]]
        diff_node_prob = node_scores[diff_node[:,0], diff_node[:,1]]
        hidden_new[bool_diff_node_idx][:] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)
        
        # extract sampled nodes an their embeddings
        new_nodes  = nodes[bool_same_node_idx]
        hidden_new = hidden_new[bool_same_node_idx]
        # ============ node sampling ends ============

        return hidden_new, new_nodes, bool_same_node_idx

class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        print(params)
        self.n_layer     = params.n_layer
        self.hidden_dim  = params.hidden_dim
        self.attn_dim    = params.attn_dim
        self.n_ent       = params.n_ent
        self.n_rel       = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader      = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act  = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.n_ent, \
                                            MESS_FUNC=params.MESS_FUNC, AGG_FUNC=params.AGG_FUNC, COMB_FUNC=params.COMB_FUNC, \
                                            n_node_topk=i_n_node_topk, n_edge_topk=self.n_edge_topk, tau=params.tau, act=act))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)       
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate    = nn.GRU(self.hidden_dim, self.hidden_dim)

        # calculate parameters
        num_parameters = sum(p.numel() for p in self.parameters())
        print('==> num_parameters: {}'.format(num_parameters))

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):
        def freeze(m):
            m.requires_grad=False
        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def forward(self, subs, rels, mode='train', recordFullTrace=False, recordDistance=False):
        n      = len(subs)                                                                # n == B (Batchsize)
        q_sub  = torch.LongTensor(subs).cuda()                                            # [B]
        q_rel  = torch.LongTensor(rels).cuda()                                            # [B]
        h0     = torch.zeros((1, n, self.hidden_dim)).cuda()                              # [1, B, dim]
        nodes  = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)  # [B, 2] with (batch_idx, node_idx)
        hidden = torch.zeros(n, self.hidden_dim).cuda()                                   # [B, dim]
        avg_nodes_list, avg_edges_list = [], []
        if recordFullTrace: self.fullTraceResults = torch.zeros((n, self.n_ent))
        
        
        for i in range(self.n_layer):
            # print(f'layer {i}, n_node_topk = {self.n_node_topk[i]}')s
            if self.n_node_topk[i] > 0:
                # layers with sampling
                # nodes (of i-th layer): [k1, 2]
                # edges (of i-th layer): [k2, 6]
                # old_nodes_new_idx (of previous layer): [k1']
                nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), n, mode=mode)
                n_node  = nodes.size(0)
                # GNN forward -> get hidden representation at i-th layer
                # hidden: [k1, dim]
                hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, n)
                # combine h0 and hi -> update hi with gate operation
                h0          = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
                h0          = h0[0, sampled_nodes_idx, :].unsqueeze(0)
                hidden      = self.dropout(hidden)
                hidden, h0  = self.gate(hidden.unsqueeze(0), h0)
                hidden      = hidden.squeeze(0)
            else:
                # sampled edges w.r.t. sampled_nodes_idx
                if i == self.n_node_topk.index(-1):
                    int_edge_heads = sampled_nodes_idx[edges[:,4]]
                    int_edge_tails = sampled_nodes_idx[edges[:,5]]
                    bool_edge = (int_edge_heads * int_edge_tails).bool()
                    sampled_edges = edges[bool_edge][:,:4]
                    head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
                    tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)
                    edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

                # layers without sampling
                hidden      = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, n, last_batchidx_count)
                hidden      = self.dropout(hidden)
                hidden, h0  = self.gate(hidden.unsqueeze(0), h0)
                hidden      = hidden.squeeze(0)

            avg_nodes_list.append(hidden.shape[0] / n)
            avg_edges_list.append(edges.shape[0]  / n)

            # record with nodes in each batch
            if recordFullTrace:
                self.fullTraceResults[nodes[:,0], nodes[:,1]] += 1
                for (batch_idx, node_idx) in nodes.data.cpu():
                    self.fullTraceResults[batch_idx].append(node_idx)

        # source to all visited nodes -> distance distribution
        if mode == 'valid' and recordDistance:
            self.batch_distance = defaultdict(list)
            for batch_idx in range(n):
                this_batch_nodes = torch.where(nodes[:,0] == batch_idx)
                np_subs = q_sub[nodes[:,0][this_batch_nodes]].cpu().numpy()
                np_objs = nodes[:,1][this_batch_nodes].cpu().numpy()
                distance = self.distance_matrix[np_subs, np_objs]
                for hop in range(1, 11):
                    num = (np.where(distance == hop)[0].shape[0])
                    self.batch_distance[hop].append(num)

        # readout
        scores     = self.W_final(hidden).squeeze(-1)                   # [K, 2] (batch_idx, node_idx) K is w.r.t. n_nodes
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         # non-visited entities have 0 scores
        scores_all[[nodes[:,0], nodes[:,1]]] = scores                   # [B, n_all_nodes]

        return scores_all, (avg_nodes_list, avg_edges_list)
