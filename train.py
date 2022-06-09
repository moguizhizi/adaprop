import random
import os
import argparse
import torch
import time
import pickle as pkl
import numpy as np

from load_data import DataLoader
from base_model import BaseModel
from utils import *
from config import *
from base_HPO import RF_HPO

parser = argparse.ArgumentParser(description="Parser for PRINCE")
parser.add_argument('--data_path', type=str, default='data/family/')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--topk', type=int, default=-1)
parser.add_argument('--layers', type=int, default=-1)
parser.add_argument('--sampling', type=str, default='incremental')
parser.add_argument('--HPO_acq', type=str, default='max')
parser.add_argument('--weight', type=str, default=None)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--loss_in_each_layer', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--saveWeight', action='store_true')
parser.add_argument('--useSearchLog', action='store_true')
parser.add_argument('--candidate', type=str, default=None)
args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    opts = args
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    torch.cuda.set_device(opts.gpu)
    print('==> gpu:', opts.gpu)
    loader = DataLoader(opts.data_path)
    opts.dataset = dataset
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.date  = str(time.asctime(time.localtime(time.time())))
    
    # check all output paths
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{loader.task_dir}/saveModel/')
    HPO_save_path = f'./results/{dataset}/search_log.pkl'

    def run_model(params, saveWeight=args.saveWeight, save_path=HPO_save_path, opts=opts, loader=loader):
        # params -> opts
        opts.lr = params['lr']
        opts.decay_rate = params['decay_rate']
        opts.lamb = params['lamb']
        opts.hidden_dim = params['hidden_dim']
        opts.attn_dim = params['attn_dim']
        opts.n_layer = params['n_layer']
        opts.dropout = params['dropout']
        opts.act = params['act']
        opts.MESS_FUNC = params['MESS_FUNC']
        opts.AGG_FUNC = params['AGG_FUNC']
        opts.COMB_FUNC = params['COMB_FUNC']
        opts.n_batch = params['n_batch']
        opts.n_tbatch = params['n_tbatch']
        opts.n_node_topk = [params['n_node_topk']] * params['n_layer']
        opts.n_edge_topk = params['n_edge_topk']

        # build model w.r.t. opts
        model = BaseModel(opts, loader)
        opts_str = str(opts)
        
        # train model
        best_mrr = 0
        best_test_mrr = 0
        val_eval_dict, test_eval_dict = {}, {}
        time_begin = time.time()
        for epoch in range(opts.max_epoch):
            v_mrr, v_h1, v_h10 = model.train_batch()
            val_eval_dict[epoch+1] = (v_mrr, v_h1, v_h10)
            
            if v_mrr > best_mrr:
                
                _, _, _, t_mrr, t_h1, t_h10 = model.evaluate(eval_val=False, eval_test=True)
                test_eval_dict[epoch+1] = (t_mrr, t_h1, t_h10)

                best_mrr, best_test_mrr = v_mrr, t_mrr
                best_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10)
                print(str(epoch+1) + '\t' + best_str)

                if saveWeight:
                    BestMetricStr = f'ValMRR_{str(best_mrr)[:5]}'
                    model.saveModelToFiles(BestMetricStr, deleteLastFile=True)

        # save to local file
        opts.training_time = time.time() - time_begin
        if not os.path.exists(save_path):
            HPO_records = {}
        else:
            HPO_records = pkl.load(open(save_path, 'rb'))
        HPO_records[opts_str] = (best_mrr, best_test_mrr, val_eval_dict, test_eval_dict, params, opts)
        pkl.dump(HPO_records, open(save_path, 'wb'))

        return best_mrr

    def loadSearchLog(file):
        assert os.path.exists(file)
        data = pkl.load(open(file, 'rb'))
        config_list, mrr_list = [], []
        for HP_key, HP_values in data.items():
            (best_mrr, best_test_mrr, val_eval_dict, test_eval_dict, params, opts) = HP_values
            
            # config_list.append(params)
            # mrr_list.append(best_mrr)
            for epoch, eval_res in val_eval_dict.items():
                params['epoch'] = epoch
                config_list.append(params)
                mrr_list.append(eval_res[0])

        print(f'==> load {len(config_list)} trials from file: {file}')
        return config_list, mrr_list

    
    if opts.candidate != None:
        def getNextConfig():
            data = pkl.load(open(opts.candidate, 'rb'))
            for idx in range(len(data)):
                if data[idx]['status'] == 'none':
                    data[idx]['status'] = 'running'
                    pkl.dump(data, open(opts.candidate, 'wb'))
                    return idx, data[idx]['param']
            return -1, None

        while True:
            idx, param = getNextConfig()
            print(idx, param)
            if idx == -1: break

            try:
                run_model(param)
            except:
                continue

        exit()

    # standard HPO pipeline
    HPO_search_space['epoch'] = ('choice', [opts.max_epoch])
    HPO_instance = RF_HPO(kgeModelName='PRINCE-v1', obj_function=run_model, dataset_name=opts.dataset, HP_info=HPO_search_space, acq=opts.HPO_acq)
    
    if opts.useSearchLog:
        config_list, mrr_list = loadSearchLog(HPO_save_path)
        dataset_names = [opts.dataset for i in range(len(config_list))]
        HPO_instance.pretrain(config_list, mrr_list, dataset_names=dataset_names)
    
    max_trials, sample_num = 100, 1e4
    HPO_instance.runTrials(max_trials, sample_num, explore_trials=100)

