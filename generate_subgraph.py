import random
import os
import argparse
import torch
import time
import pickle as pkl
import numpy as np
import os
import pickle as pkl                 
import argparse
import os
import sys
import inspect
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix
from load_data import DataLoader
from base_model import BaseModel
from utils import *
from config import *
from base_HPO import RF_HPO

parser = argparse.ArgumentParser(description="Parser for PRINCE")
parser.add_argument('--data_path', type=str, default='data/WN18RR/')
parser.add_argument('-sample_ratio', type=float, default=0.2)
parser.add_argument('-num_starts', type=int, default=10)


class KGGenerator_randomWalks:
    def __init__(self, all_triples, sample_ratio, num_starts, savePath, repeat_num=0, split_ratio=[0.9, 0.05, 0.05]):        
        self.split_ratio  = split_ratio
        self.sample_ratio = sample_ratio
        self.repeat_num   = repeat_num
        self.num_starts   = num_starts

        # setup graph
        homoGraph = self.triplesToNxGraph(all_triples)
        diGraph   = self.triplesToNxDiGraph(all_triples)

        # sampling via random walk
        num_nodes = homoGraph.number_of_nodes()
        target_num_nodes = int(num_nodes * self.sample_ratio)
        if num_starts == 1:
            sampled_nodes = self.random_walk_induced_graph_sampling(homoGraph, target_num_nodes)
        else:
            sampled_nodes = self.multi_starts_random_walk_induced_graph_sampling(homoGraph, target_num_nodes, num_starts)

        sampled_graph      = diGraph.subgraph(sampled_nodes)
        homo_sampled_graph = homoGraph.subgraph(sampled_nodes)

        # show num of connected parts
        connected_components = nx.connected_components(homo_sampled_graph)
        connected_subgraphs = []
        for c in connected_components:
            connected_subgraphs.append(homo_sampled_graph.subgraph(c).copy())
        print(f'==> #connected_components:{len(connected_subgraphs)}')

        # build sampled KG
        self.all_triples = []
        self.relations   = []
        self.entities    = []
        for edge in list(sampled_graph.edges(data=True)):
            h,t = edge[0], edge[1]
            r = edge[2]['relation']
            self.all_triples.append((h,r,t))
            self.relations.append(r)
            self.entities.append(h)
            self.entities.append(t)

        # assign new index to entities/relation
        self.entities  = sorted(list(set(self.entities)))
        self.nentity   = len(self.entities)
        self.relations = sorted(list(set(self.relations)))
        self.nrelation = len(self.relations)
        self.ntriples  = len(self.all_triples)
        self.sparsity  = self.ntriples / (self.nentity * self.nentity * self.nrelation)
        self.entity_mapping_dict = {}
        self.relation_mapping_dict = {}

        # print('dataset={}, nentity={}, sampled ratio={}, sparsity={}'.format(
        #     args.dataset, self.nentity, self.sample_ratio, self.sparsity))

        # key:   origin index
        # value: new assigned index
        for idx in range(self.nentity):
            self.entity_mapping_dict[self.entities[idx]] = idx
        for idx in range(self.nrelation):
            self.relation_mapping_dict[self.relations[idx]] = idx

        # get new triples via entitie_mapping_dict
        self.all_new_triples = []
        for (h,r,t) in self.all_triples:
            new_h, new_t = self.entity_mapping_dict[h], self.entity_mapping_dict[t]
            new_r = self.relation_mapping_dict[r]
            new_triples = (new_h, new_r, new_t)
            self.all_new_triples.append(new_triples)

        # shuffle triples
        random.shuffle(self.all_new_triples)

        # split and save data
        self.fact_data, self.train_data, self.valid_data, self.test_data = self.splitData()
        print(len(self.fact_data),len(self.train_data),len(self.valid_data),len(self.test_data))

        self.saveData(savePath)

    @staticmethod
    def triplesToNxGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.Graph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)
        edges = list(set([(h,t) for (h,r,t) in triples]))
        graph.add_edges_from(edges)

        return graph

    @staticmethod
    def triplesToNxDiGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.MultiDiGraph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)

        for (h,r,t) in triples:
            graph.add_edges_from([(h,t)], relation=r)

        return graph
        
    @staticmethod
    def random_walk_induced_graph_sampling(complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        Sampled_nodes = set([complete_graph.nodes[index_of_first_random_node]['id']])

        iteration   = 1
        growth_size = 2
        check_iters = 100
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node; print(f'==> curr_node: {curr_node}')
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % check_iters == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                    print(f'==> boost seaching, skip to No.{curr_node} node')
                    curr_node = random.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        # sampled_graph = complete_graph.subgraph(Sampled_nodes)
        # return sampled_graph
        return Sampled_nodes

    @staticmethod
    def multi_starts_random_walk_induced_graph_sampling(complete_graph, nodes_to_sample, num_starts):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)

        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes        = len(complete_graph.nodes())
        start_candidate = [random.randint(0, nr_nodes - 1) for i in range(num_starts)]
        Sampled_nodes   = set()

        for idx, index_of_first_random_node in enumerate(start_candidate):
            Sampled_nodes.add(complete_graph.nodes[index_of_first_random_node]['id'])
            iteration           = 1
            growth_size         = 2
            check_iters         = 100
            nodes_before_t_iter = 0
            target_num = int((idx+1) * nodes_to_sample / num_starts)
            curr_node  = index_of_first_random_node

            while len(Sampled_nodes) < target_num:
                edges = [n for n in complete_graph.neighbors(curr_node)]
                index_of_edge = random.randint(0, len(edges) - 1)
                chosen_node = edges[index_of_edge]
                Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
                curr_node = chosen_node
                iteration = iteration + 1

                if iteration % check_iters == 0:
                    if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                        print(f'==> boost seaching, skip to No.{curr_node} node')
                        curr_node = random.randint(0, nr_nodes - 1)
                    nodes_before_t_iter = len(Sampled_nodes)

        return Sampled_nodes

    def splitData(self):
        '''
            split triples with certain ratio stored in self.split_ratio
        '''
        n1 = int(self.ntriples * self.split_ratio[0])
        n1_fact = n1*3//4
        n2 = int(n1 + self.ntriples * self.split_ratio[1])

        # return self.all_new_triples[:n1], self.all_new_triples[n1:n2], self.all_new_triples[n2:]
        return self.all_new_triples[:n1_fact], self.all_new_triples[n1_fact:n1], self.all_new_triples[n1:n2], self.all_new_triples[n2:]

    def saveData(self, savePath):
        saveFolder = savePath.replace('-', '_')
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        dataDict = {}
        dataDict['nentity']    = self.nentity
        dataDict['nrelation']  = self.nrelation
        dataDict['fact_data']  = self.fact_data
        dataDict['train_data'] = self.train_data
        dataDict['valid_data'] = self.valid_data
        dataDict['test_data']  = self.test_data
        
        dataDict['entity_mapping_dict'] = self.entity_mapping_dict
        dataDict['relation_mapping_dict'] = self.relation_mapping_dict
        
        dictPath = os.path.join(saveFolder, 'dataset.pkl')
        print('==> save to:', dictPath); 
        # exit()
        pkl.dump(dataDict, open(dictPath, "wb" ))


if __name__ == '__main__':
    opts = parser.parse_args()
    dataset = opts.data_path.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    loader = DataLoader(opts.data_path)

    # all the observed triples (training triples) to be sampled
    train_triples = loader.fact_triple + loader.train_triple
    savePath = f'data/sampled_{dataset}_{opts.sample_ratio}_start_{opts.num_starts}/'
    print(len(train_triples), savePath)

    # exit()
    generator = KGGenerator_randomWalks(train_triples, sample_ratio=opts.sample_ratio, \
                num_starts=opts.num_starts, savePath=savePath) 


