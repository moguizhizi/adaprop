import torch
import numpy as np
import time
import os

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from models import GNNModel
from utils import *
from tqdm import tqdm
from collections import defaultdict

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = GNNModel(args, loader)
        self.model.cuda()

        self.args = args
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, verbose=True)

        # training recorder
        self.t_time = 0
        self.lastSaveGNNPath = None

        self.modelName = f'{args.n_layer}-layers'
        for i in range(args.n_layer):
            i_n_node_topk = args.n_node_topk if 'int' in str(type(args.n_node_topk)) else args.n_node_topk[i]
            self.modelName += f'-{i_n_node_topk}'
        print(f'==> model name: {self.modelName}')

    def saveModelToFiles(self, best_metric, deleteLastFile=True):
        savePath = f'{self.loader.task_dir}/saveModel/{self.modelName}-{best_metric}.pt'
        print(f'Save checkpoint to : {savePath}')
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mrr':best_metric,
                }, savePath)

        if deleteLastFile and self.lastSaveGNNPath != None:
            print(f'Remove last checkpoint: {self.lastSaveGNNPath}')
            os.remove(self.lastSaveGNNPath)

        self.lastSaveGNNPath = savePath

    def loadModel(self, filePath, layers=-1):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device(f'cuda:{self.args.gpu}'))

        if layers != -1:
            extra_layers = self.model.gnn_layers[layers:]
            self.model.gnn_layers = self.model.gnn_layers[:layers]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.gnn_layers += extra_layers
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train_batch(self,):
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)
        avg_nodes_dict, avg_edges_dict = defaultdict(list), defaultdict(list)
        reach_tails_list = []

        t_time = time.time()
        self.model.train()
        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(self.loader.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()

            scores, (avg_nodes_list, avg_edges_list) = self.model(triple[:,0], triple[:,1])
            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:,2]).cuda()]]

            # cover tail entity or not
            reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
            reach_tails_list += reach_tails

            max_n = torch.max(scores, 1, keepdim=True)[0]

            # ori loss
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 
            # loss with cover ratio
            # loss = - pos_scores + max_n.squeeze(-1) + torch.log(torch.sum(torch.exp(scores - max_n),1))
            # loss[pos_scores == 0] *= 10
            # loss = torch.sum(loss)
            loss.backward()
            self.optimizer.step()

            for layer in range(len(avg_nodes_list)):
                avg_nodes_dict[layer].append(avg_nodes_list[layer])
                avg_edges_dict[layer].append(avg_edges_list[layer])

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()

        self.t_time += time.time() - t_time

        # verbose
        print(f'[train] covering tail ratio: {1 - sum(reach_tails_list) / len(reach_tails_list)}')
        for layer in sorted(avg_nodes_dict.keys()):
            print(f'[train] layer: {layer+1}, Avg nodes: {np.mean(avg_nodes_dict[layer])}, Avg edges: {np.mean(avg_edges_dict[layer])}.')

        # evaluate on validate set
        v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10 = self.evaluate(eval_val=True, eval_test=False)
        self.scheduler.step(v_mrr)
        self.loader.shuffle_train()

        return v_mrr, v_h1, v_h10

    def evaluate(self, verbose=False, eval_val=True, eval_test=False, recordDistance=False):
        self.model.eval()
        batch_size = self.n_tbatch

        # - - - - - - val set - - - - - - 
        if not eval_val:
            v_mrr, v_h1, v_h10 = -1, -1, -1
        else:
            n_data = self.n_valid
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []
            val_avg_nodes_dict, val_avg_edges_dict = defaultdict(list), defaultdict(list)
            val_distance_dict = defaultdict(list)
            val_reach_tails_list = []
            i_time = time.time()
            val_mrr_with_query = []

            # - - - - - - val set - - - - - - 
            iterator = tqdm(range(n_batch)) if verbose else range(n_batch)
            for i in iterator:
                start = i*batch_size
                end = min(n_data, (i+1)*batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
                scores, (avg_nodes_list, avg_edges_list) = self.model(subs, rels, mode='valid')
                scores = scores.data.cpu().numpy()

                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent, ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot) 
                filters = np.array(filters)

                # scores / objs / filters: [batch_size, n_ent]
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

                for layer in range(len(avg_nodes_list)):
                    val_avg_nodes_dict[layer].append(avg_nodes_list[layer])
                    val_avg_edges_dict[layer].append(avg_edges_list[layer])

                # cover tails or not
                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                val_reach_tails_list += reach_tails

                # for idx in range(len(ranks)):
                #     batch_idx, ent_indx = int(ans[0][idx]), int(ans[1][idx])
                #     query_and_rank = [subs[batch_idx], objs[batch_idx, ent_indx], ranks[idx]]
                #     val_mrr_with_query.append(query_and_rank)

                if recordDistance:
                    batch_distance = self.model.batch_distance
                    for hop, nums in batch_distance.items():
                        val_distance_dict[hop] += nums

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h10 = cal_performance(ranking)

            if verbose:
                print(f'[val]  covering tail ratio: {len(val_reach_tails_list)}, {1 - sum(val_reach_tails_list) / len(val_reach_tails_list)}')
                for layer in sorted(val_avg_nodes_dict.keys()):
                    print(f'[val]  layer: {layer+1}, Avg nodes: {np.mean(val_avg_nodes_dict[layer])}, Avg edges: {np.mean(val_avg_edges_dict[layer])}.')
            
            if recordDistance:
                total = 0
                for hop, data in val_distance_dict.items():
                    total += sum(data)
                for hop, data in val_distance_dict.items():
                    print(hop, len(data), sum(data), sum(data)/total)

        # - - - - - - test set - - - - - - 
        if not eval_test:
            t_mrr, t_h1, t_h10 = -1, -1, -1
        else:
            n_data = self.n_test
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []
            self.model.eval()
            test_avg_nodes_dict, test_avg_edges_dict = defaultdict(list), defaultdict(list)
            test_reach_tails_list = []
            test_mrr_with_query = []

            iterator = tqdm(range(n_batch)) if verbose else range(n_batch)
            for i in iterator:
                start = i*batch_size
                end = min(n_data, (i+1)*batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
                scores, (avg_nodes_list, avg_edges_list) = self.model(subs, rels, mode='test')
                scores = scores.data.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent, ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

                for layer in range(len(avg_nodes_list)):
                    test_avg_nodes_dict[layer].append(avg_nodes_list[layer])
                    test_avg_edges_dict[layer].append(avg_edges_list[layer])

                # cover tails or not
                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                test_reach_tails_list += reach_tails

                # for idx in range(len(ranks)):
                #     batch_idx, ent_indx = int(ans[0][idx]), int(ans[1][idx])
                #     query_and_rank = [subs[batch_idx], objs[batch_idx, ent_indx], ranks[idx]]
                #     test_mrr_with_query.append(query_and_rank)

            ranking = np.array(ranking)
            t_mrr, t_h1, t_h10 = cal_performance(ranking)
            # test_mrr_with_query = np.array(test_mrr_with_query)
            # savePath = f'{self.loader.task_dir}/saveModel/TEST_query_and_rank.npy'
            # np.save(open(savePath, 'wb'), test_mrr_with_query)
            
            # !!!
            import pickle
            with open(os.path.join(self.args.data_path, 'trank.pkl'), 'wb') as fp:
                pickle.dump(list(ranking), fp)

            if verbose:
                print(f'[test] covering tail ratio: {len(test_reach_tails_list)}, {1 - sum(test_reach_tails_list) / len(test_reach_tails_list)}')
                for layer in sorted(test_avg_nodes_dict.keys()):
                    print(f'[test] layer: {layer+1}, Avg nodes: {np.mean(test_avg_nodes_dict[layer])}, Avg edges: {np.mean(test_avg_edges_dict[layer])}.')

        # i_time = time.time() - i_time
        # out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
        
        return v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10

    def evaluate_with_cover_ratio(self, ):
        batch_size = self.n_tbatch
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        val_avg_nodes_dict, val_avg_edges_dict = defaultdict(list), defaultdict(list)
        val_distance_dict = defaultdict(list)
        val_reach_tails_list = []
        val_visit_rank_list = []
        i_time = time.time()

        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')

            scores, (avg_nodes_list, avg_edges_list) = self.model(subs, rels, mode='valid', recordFullTrace=True)
            scores = scores.data.cpu().numpy()

            filters = []
            for i in range(len(subs)):
                filt = self.loader.filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot) 
            filters = np.array(filters)

            # scores / objs / filters: [batch_size, n_ent]
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks

            for layer in range(len(avg_nodes_list)):
                val_avg_nodes_dict[layer].append(avg_nodes_list[layer])
                val_avg_edges_dict[layer].append(avg_edges_list[layer])

            # cover tails or not
            ans = np.nonzero(objs)
            ans_score = scores[ans].reshape(-1)
            reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
            val_reach_tails_list += reach_tails

            # trace visit nodes
            fullTraceResults = torch.argsort(self.model.fullTraceResults, descending=True)
            for batch_idx in range(len(fullTraceResults)):
                this_batch_obj = np.nonzero(objs[batch_idx])[0].tolist()
                if len(this_batch_obj) > 0:
                    # visit_trace = uniqueWithoutSort(fullTraceResults[batch_idx])
                    visit_trace = fullTraceResults[batch_idx].tolist()
                    for obj in this_batch_obj:
                        val_visit_rank_list.append(visit_trace.index(obj) + 1)

        ranking = np.array(ranking)
        v_mrr, v_h1, v_h10 = cal_performance(ranking)

        # verbose
        print('[VALID] MRR:%.4f H@1:%.4f H@10:%.4f \n'%(v_mrr, v_h1, v_h10))
        
        print(f'[VALID]  covering tail ratio: {len(val_reach_tails_list)}, {1 - sum(val_reach_tails_list) / len(val_reach_tails_list)}')
        for layer in sorted(val_avg_nodes_dict.keys()):
            print(f'[VALID]  layer: {layer+1}, Avg nodes: {np.mean(val_avg_nodes_dict[layer])}, Avg edges: {np.mean(val_avg_edges_dict[layer])}.')
        
        val_visit_rank_list = np.array(val_visit_rank_list)
        for rank_thres in [10, 100, 500, 1000, 3000]:
            ratio = np.where(val_visit_rank_list <= rank_thres)[0].shape[0] / len(val_visit_rank_list)
            print(f'[VALID] rank_thres={rank_thres}, ratio={ratio}')

        return val_visit_rank_list

