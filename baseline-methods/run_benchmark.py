#!/usr/bin/python

import sys
import os
import getopt
import pickle
import pandas as pd

from benchmark import Benchmark
from DataLoader import *

sys.path.append('../data/HSLS')
from hsls_utils import *

sys.path.append('../FACT')
from FACT.postprocess import *

sys.path.append('../leveraging-python')
from utils import leveraging_approach

import warnings
warnings.filterwarnings("ignore")


def main(argv):
    model = 'gbm'
    fair = 'reduction'
    seed = 42
    constraint = 'eo'
    num_iter = 10
    inputfile = 'enem'
    
    try:
        opts, args = getopt.getopt(argv,"hm:s:f:c:n:i:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Correct Usage:\n')
            print('python run_benchmark.py -m [model name] -f [fair method] -c [constraint] -n [num iter] -i [inputfile] -s [seed]')
            print('\n')
            print('Options for arguments:')
            print('[model name]: gbm, logit, rf (Default: gbm)')
            print('[fair method]: reduction, eqodds, roc (Default: reduction)')
            print('[constraint]: eo, sp, (Default: eo)')
            print('[num iter]: Any positive integer (Default: 10) ')
            print('[inputfile]: hsls, enem-20000, enem-50000, ...  (Default: hsls)')
            print('[seed]: Any integer (Default: 42)')
            print('\n')
            sys.exit()
        elif opt == "-m":
            model = arg
        elif opt  == "-s":
            seed = int(arg)
        elif opt == '-f':
            fair = arg
        elif opt == '-c':
            constraint = arg
        elif opt == '-n':
            num_iter = int(arg)
        elif opt == '-i':
            inputfile = arg
        
        
    path = '../data/'
    if inputfile == 'hsls':
        file = 'HSLS/hsls_knn_impute.pkl'
        df = load_hsls_imputed(path, file, [])
        privileged_groups = [{'racebin': 1}]
        unprivileged_groups = [{'racebin': 0}]
        protected_attrs = ['racebin']
        label_name = 'gradebin'
        
    elif inputfile == 'enem':
        df = pd.read_pickle(path+'ENEM/enem-50000-20.pkl')
        privileged_groups = [{'racebin': 1}]
        unprivileged_groups = [{'racebin': 0}]
        protected_attrs = ['racebin']
        label_name = 'gradebin'
        df[label_name] = df[label_name].astype(int)
        
    elif inputfile == 'adult':
        df = load_data('adult')
        privileged_groups = [{'gender': 1}]
        unprivileged_groups = [{'gender': 0}]
        protected_attrs = ['gender']
        label_name = 'income'

    elif inputfile == 'adult_FATO':
        df = load_data('adult', modified = True)
        privileged_groups = [{'gender': 1}]
        unprivileged_groups = [{'gender': 0}]
        protected_attrs = ['gender']
        label_name = 'income'
        
        
    elif inputfile == 'compas':
        df = load_data('compas')
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        protected_attrs = ['race']
        label_name = 'is_recid'

    elif inputfile == 'synth2':
        file = 'synthetic/gaussian_synth2.pkl'
        filename = path + file

        df = pd.read_pickle(filename)  
        # def sample_from_gaussian(pos_mean,
        #                         pos_cov,
        #                         neg_mean,
        #                         neg_cov,
        #                         thr=0,
        #                         n_pos=200,
        #                         n_neg=200,
        #                         seed=0,
        #                         corr_sens=True):
        #     np.random.seed(seed)
        #     x_pos = np.random.multivariate_normal(pos_mean, pos_cov, n_pos)
        #     np.random.seed(seed)
        #     x_neg = np.random.multivariate_normal(neg_mean, neg_cov, n_neg)
        #     X = np.vstack((x_pos, x_neg))
        #     y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
        #     n = y.shape[0]
        #     if corr_sens:
        #         # correlated sens data
        #         sens_attr = np.zeros(n)
        #         idx = np.where(X[:,0] > thr)[0]
        #         sens_attr[idx] = 1
        #     else:
        #         # independent sens data
        #         np.random.seed(seed)
        #         sens_attr = np.random.binomial(1, 0.5, n)
        #     return X, y, sens_attr

        # ## NOTE change these variables for different distribution/generation of synth data.
        # pos_mean = np.array([2,2])
        # pos_cov = np.array([[5, 1], [1,5]])
        # neg_mean = np.array([-2,-2])
        # neg_cov = np.array([[10, 1],[1, 3]])
        # n_pos = 1000
        # n_neg = 600
        # thr = 0
        # corr_sens = True
        # X, y, sens = sample_from_gaussian(pos_mean,
        #                                     pos_cov,
        #                                     neg_mean,
        #                                     neg_cov,
        #                                     thr=thr,
        #                                     n_pos=n_pos,
        #                                     n_neg=n_neg,
        #                                     corr_sens=corr_sens)
        # X = np.concatenate((X, np.expand_dims(sens, 1)), axis=1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        # dtypes = None
        # dtypes_ = None
        # sens_idc = [2]
        # X_train_removed = X_train[:,:2]
        # X_test_removed = X_test[:,:2]
        # race_idx = None
        # sex_idx = None

        # df_array = np.hstack([X_train, y_train.reshape(-1, 1)])
        # df = pd.DataFrame(df_array)
        # df = df.rename(columns={0: "x0", 1: "x1", 2: "s", 3: "y"})

        privileged_groups = [{'s': 0}]
        unprivileged_groups = [{'s': 1}]
        protected_attrs = ['s']
        label_name = 'y'

        
    else: 
        print('Invalid Input Dataset Name')
        sys.exit(2)

    print('#### Data Loaded. ')
    

    #### Setting group attribute and label ####
    
    
    bm = Benchmark(df, privileged_groups, unprivileged_groups, protected_attrs,label_name)


    #### Run benchmarks ####
    if fair == 'reduction':
        # eps_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2] # Epsilon values for reduction method #
        # eps_list = [ 0.1,0.5, 1, 2, 5, 10] 
        #eps_list = [ 0.1, 0.5, 1, 2, 5, 10, 15] 
        eps_list = [100, 150， 200, 30, 35, 40, 45, 50]
        if constraint == 'sp':
            results = bm.reduction(model, num_iter, seed, params=eps_list, constraint='DemographicParity')
        elif constraint == 'eo':
            results = bm.reduction(model, num_iter, seed, params=eps_list, constraint='EqualizedOdds')
        
    elif fair == 'eqodds':
        results = bm.eqodds(model, num_iter, seed)
        constraint = ''
    
    elif fair == 'caleqodds': 
        results = bm.eqodds(model, num_iter, seed, calibrated=True, constraint=constraint)
        constraint = ''
        
    elif fair == 'roc':
        eps_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2] # Metric ub and lb values for roc method #
        if constraint == 'sp':
            results = bm.roc(model, num_iter, seed, params=eps_list, constraint='DemographicParity')
        elif constraint == 'eo':
            results = bm.roc(model, num_iter, seed, params=eps_list, constraint='EqualizedOdds')

    elif fair == 'fact':
        results = post_process(model, inputfile)
        
    elif fair == 'leveraging':
        _, results, _ = leveraging_approach(df, protected_attrs, label_name, use_protected=True, model = model, num_iter = num_iter, rand_seed =seed)
        
    elif fair == 'original':
        results = bm.original(model, num_iter, seed)
        constraint = ''
        
    else:
        print('Undefined method')
        sys.exit(2)


    result_path = './adult_FATO/'
    #result_path = './adult/'
    filename = fair+'_'+model+'_s'+str(seed)+'_' + constraint+'_additional.pkl'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path+filename, 'wb+') as f: 
        pickle.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1:])
