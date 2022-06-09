HPO_search_space = {
        # discrete
        'lr':                    ('choice', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), 
        'hidden_dim':            ('choice', [16, 32, 48, 64, 128, 256]),
        'attn_dim':              ('choice', [2, 4, 8, 16, 32, 64]),
        'n_layer':               ('choice', [3, 4, 5, 6, 7, 8]),
        'n_node_topk':           ('choice', [100, 200, 500, 1000, 2000]),
        'n_edge_topk':           ('choice', [-1]),
        'act':                   ('choice', ['relu', 'idd', 'tanh']),
        'MESS_FUNC':             ('choice', ['TransE', 'DistMult', 'RotatE']),
        'AGG_FUNC':              ('choice', ['sum', 'mean', 'max']),
        'COMB_FUNC':             ('choice', ['discard', 'residual']),
        'n_batch':               ('choice', [8, 16, 32, 64, 128]),
        'n_tbatch':              ('choice', [16]),
        
        # continuous
        'decay_rate':            ('uniform', (0.8, 1)),
        'lamb':                  ('uniform', (1e-5, 1e-3)),
        'dropout':               ('uniform', (0, 0.5)),
    }