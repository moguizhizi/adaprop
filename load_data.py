import os
import torch
import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir):
        self.task_dir = task_dir

        if 'sampled' in self.task_dir:
            dataDict = pkl.load(open(os.path.join(task_dir, 'dataset.pkl'), 'rb'))
            self.n_ent = dataDict['nentity']   
            self.n_rel = dataDict['nrelation'] 
            self.fact_triple = dataDict['fact_data'] 
            self.train_triple = dataDict['train_data']
            self.valid_triple = dataDict['valid_data']
            self.test_triple = dataDict['test_data'] 
            self.filters = defaultdict(lambda:set())
            self.addToFilter(self.fact_triple)
            self.addToFilter(self.train_triple)
            self.addToFilter(self.valid_triple)
            self.addToFilter(self.test_triple)

        else:
            with open(os.path.join(task_dir, 'entities.txt')) as f:
                self.entity2id = dict()
                n_ent = 0
                for line in f:
                    entity = line.strip()
                    self.entity2id[entity] = n_ent
                    n_ent += 1

            with open(os.path.join(task_dir, 'relations.txt')) as f:
                self.relation2id = dict()
                n_rel = 0
                for line in f:
                    relation = line.strip()
                    self.relation2id[relation] = n_rel
                    n_rel += 1

            self.n_ent = n_ent
            self.n_rel = n_rel
            self.filters = defaultdict(lambda:set())
            self.fact_triple  = self.read_triples('facts.txt')
            self.train_triple = self.read_triples('train.txt')
            self.valid_triple = self.read_triples('valid.txt')
            self.test_triple  = self.read_triples('test.txt')
    
        # add inverse
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data  = self.double_triple(self.test_triple)

        self.load_graph(self.fact_data)
        self.all_observed_data = self.double_triple(self.fact_triple)+self.double_triple(self.train_triple)
        self.load_test_graph(self.all_observed_data)

        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_a  = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_ent:', self.n_ent, 'n_rel:', self.n_rel)
        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def addToFilter(self, triples):
        for (h,r,t) in triples:
            self.filters[(h,r)].add(t)
            self.filters[(t,r+self.n_rel)].add(h)
        return

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples

    def load_graph(self, triples):
        # (e, r', e)
        # r' = 2 * n_rel, r' is manual generated and not exist in the original KG
        # self.KG: shape=(self.n_fact, 3)
        # M_sub shape=(self.n_fact, self.n_ent), store projection from head entity to triples
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        # triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, batchsize, mode='train'):
        if mode == 'train':
            KG    = self.KG
            M_sub = self.M_sub
        else:
            KG    = self.tKG
            M_sub = self.tM_sub

        # nodes: [N_ent_of_all_batch_last, 2] with (batch_idx, node_idx)
        # [N_ent, N_ent_of_all_batch_last]
        node_1hot     = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
        # [N_fact, N_ent] * [N_ent, N_ent_of_all_batch_last] -> [N_fact, N_ent_of_all_batch_last]
        edge_1hot     = M_sub.dot(node_1hot)
        # [2, N_edge_of_all_batch] with (fact_idx, batch_idx)
        edges         = np.nonzero(edge_1hot)
        # {batch_idx} + {head, rela, tail} -> concat -> [N_edge_of_all_batch, 4] with (batch_idx, head, rela, tail)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # indexing nodes | within/out of a batch | relative index
        # note that node_idx is the absolute nodes idx in original KG
        # head_nodes: [N_ent_of_all_batch_last, 2] with (batch_idx, node_idx)
        # tail_nodes: [N_ent_of_all_batch_this, 2] with (batch_idx, node_idx)
        # head_index: [N_edge_of_all_batch] with relative node idx
        # tail_index: [N_edge_of_all_batch] with relative node idx
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        # [N_edge_of_all_batch, 4] -> [N_edge_of_all_batch, 6] with (batch_idx, head, rela, tail, head_index, tail_index)
        # node that the head_index and tail_index are of this layer
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        
        # get new index for nodes in last layer
        mask = sampled_edges[:,2] == (self.n_rel*2)
        # old_nodes_new_idx: [N_ent_of_all_batch_last]
        old_nodes_new_idx = tail_index[mask].sort()[0]

        '''
        # old_nodes_with_batchidx: [N_ent_of_all_batch_last, 2] (batch_idx, node_idx)
        old_nodes_with_batchidx = sampled_edges[mask][:, [0,1]]

        # absolute index
        abs_node_index = torch.zeros((batchsize, self.n_ent)).long().cuda()
        abs_node_index[tail_nodes[:,0], tail_nodes[:,1]] = 1

        # relative index
        rel_node_index = torch.zeros((batchsize, self.n_ent)).long().cuda()
        rel_node_index[:, :] = -1
        rel_node_index[sampled_edges[:,0], sampled_edges[:,3]] = tail_index

        return tail_nodes, sampled_edges, old_nodes_new_idx, old_nodes_with_batchidx, abs_node_index, rel_node_index
        '''

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return np.array(self.train_data)[batch_idx]
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)

        subs = []
        rels = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self,):
        fact_triple = np.array(self.fact_triple)
        train_triple = np.array(self.train_triple)
        all_triple = np.concatenate([fact_triple, train_triple], axis=0)
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]
        
        self.fact_data = self.double_triple(all_triple[:n_all*3//4].tolist())
        self.train_data = np.array(self.double_triple(all_triple[n_all*3//4:].tolist()))
        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)

