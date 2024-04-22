import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer, BaseTrainer, CrossTransformerTrainer
import evaluation
from utils import check_convention, config_path_prefix
import argparse

DEFAULT_RES_SAVE_PATH_PREFIX = "./res_wish_ablation_with_number"

def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader,unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader,test_loader)
    eval_res = trainer.final_eval(test_loader)
    
    start_eval_time = time.time()
    # pred_all, y_all = trainer.test(all_loader)
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print("eval time is %s" % eval_time) 
    recorder.record_time(eval_time)
    # pred_matrix = dataloader.reconstruct_pred(pred_all)


    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    # recorder.record_pred(pred_matrix)

    return recorder

def train_convention_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    trainX, trainY, testX, testY, allX = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(trainX, trainY)
    eval_res = trainer.final_eval(testX, testY)
    pred_all = trainer.test(allX)
    pred_matrix = dataloader.reconstruct_pred(pred_all)

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)

    return recorder 




include_path = [
    'indian_cross_param_use.json',
    # 'pavia_cross_param_use.json',
    # 'WH_cross_param_use.json',

    # "indian_ssftt.json",
    # "pavia_ssftt.json",
    # 'WH_ssftt.json',

    # 'indian_casst.json',
    # 'pavia_casst.json',
    # 'WH_casst.json',

    # 'indian_SSRN.json',
    # 'pavia_SSRN.json',
    # 'WH_SSRN.json',


    # for batch process 
    # 'temp.json'
]
def run_one(param):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    name = param['net']['trainer']
    convention = check_convention(name)
    uniq_name = param.get('uniq_name', name)
    print('start to train %s...' % uniq_name)
    if convention:
        train_convention_by_param(param)
    else:
        train_by_param(param)
    print('model eval done of %s...' % uniq_name)
    path = '%s/%s' % (save_path_prefix, uniq_name) 
    recorder.to_file(path)


def run_all():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        convention = check_convention(name)
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        uniq_name = param.get('uniq_name', name)
        print('start to train %s...' % uniq_name)
        if convention:
            train_convention_by_param(param)
        else:
            train_by_param(param)
        print('model eval done of %s...' % uniq_name)
        path = '%s/%s' % (save_path_prefix, uniq_name) 
        recorder.to_file(path)

def run_one_by_args(args):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    name = args.conf
    convention = check_convention(name)
    path_param = '%s/%s' % (config_path_prefix, name)
    with open(path_param, 'r') as fin:
        param = json.loads(fin.read())

    times = args.times
    sample_num = args.sample_num
    data_sign = param['data']['data_sign']
    param['data']['data_file'] = '%s_%s' % (data_sign, sample_num) 
    uniq_name = param.get('uniq_name', name)
    uniq_name = '%s_%s_%s' % (uniq_name, sample_num, times)

    if result_file_exists(DEFAULT_RES_SAVE_PATH_PREFIX, uniq_name):
        print('%s has been run. skip...' % uniq_name)
        return

    print('start to train %s...' % uniq_name)
    if convention:
        train_convention_by_param(param)
    else:
        train_by_param(param)
    print('model eval done of %s...' % uniq_name)
    path = '%s/%s' % (save_path_prefix, uniq_name) 
    recorder.to_file(path)

def modify_pca(json_str, pca):
    json_str['data']['pca'] = pca
    json_str['data']['spectral_size'] = pca
    return json_str

def modify_layer(json_str, layer):
    json_str['net']['depth'] = layer
    return json_str

def modify_heads(json_str, heads):
    json_str['net']['heads'] = heads
    return json_str

def modify_patch_size(json_str, patch_size):
    json_str['data']['patch_size'] = patch_size
    return json_str

def run_one_multi_times(json_str, ori_uniq_name):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    times = 6
    for i in range(times): 
        uniq_name = '%s_%s' % (ori_uniq_name, i)
        if result_file_exists(DEFAULT_RES_SAVE_PATH_PREFIX, uniq_name):
            print('%s has been run. skip...' % uniq_name)
            continue

        print('start to train %s...' % uniq_name)
        train_by_param(json_str)
        print(json_str)
        print('model eval done of %s...' % uniq_name)
        path = '%s/%s' % (save_path_prefix, uniq_name) 
        recorder.to_file(path)

def run_pca():
    # pca
    def get_channel(name):
        if 'indian' in name:
            return 200
        elif 'pavia' in name:
            return 102
        elif 'WH' in name:
            return 270
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
            for p in [5, 10, 30, 50, 100, 150, 200, 250, 300]:
                spectral_num = get_channel(name)
                use_p = min(spectral_num, p)
                uniq_name = '%s_pca_%s' % (name, use_p)
                param = modify_pca(param, use_p)
                run_one_multi_times(param, uniq_name)

def run_patch_size():
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
            for patch_size in [5, 9, 13, 17, 21, 25]:
                uniq_name = '%s_patch_%s' % (name, patch_size)
                param = modify_patch_size(param, patch_size)
                run_one_multi_times(param, uniq_name)


def run_layer_head():
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
            for l in [1,2,3,4,5]:
                uniq_name = '%s_layer_%s' % (name, l)
                param = modify_layer(param, l)
                run_one_multi_times(param, uniq_name)

def run_ablation():
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
            for l in [0,1,2,3,4,5,6]:
                for num in [5, 10, 15, 20, 25, 30, 50, 70]:
                    uniq_name = '%s_ablation_%s_%s' % (name, l, num)
                    if l in [3,5,6]:
                        param['data']['random_rotate'] = False
                    param['net']['model_type'] = l 
                    param['data']['data_file'] = "%s_%s" % (param['data']['data_sign'], num)
                    run_one_multi_times(param, uniq_name)

def run_sample_num_40_50():
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
            for num in [40, 50]:
                uniq_name = '%s_%s' % (name, num)
                param['data']['data_file'] = "%s_%s" % (param['data']['data_sign'], num)
                run_one_multi_times(param, uniq_name)




def run_temp(args):
    nums = 5
    for conf in include_path:
        for sample_num in [10]:
        # for sample_num in [10]:
            for i in range(nums):
                args.conf = conf
                args.sample_num = sample_num
                args.times = i
                run_one_by_args(args)
                print('run %s_%s_%s done...' % (conf, sample_num, i))

def result_file_exists(prefix, file_name_part):
    ll = os.listdir(prefix)
    for l in ll:
        if file_name_part in l:
            return True
    return False




if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='args')
    # parser.add_argument('-n', '--nums', default=1 ,help='nums of run')
    # parser.add_argument('-p', '--patch_size', default=13 ,help='patch_size')
    # parser.add_argument('-c', '--conf', default='indian_cross_param_use.json' ,help='conf file name')
    # parser.add_argument('-t', '--times', default=1 ,help='the times of run')
    # parser.add_argument('-s', '--sample_num', default=10 ,help='sample num')
    # args = parser.parse_args()
    # run_temp(args)
    

    # run_all()
    

    # run_patch_size()
    # run_layer_head()
    # run_pca()

    run_ablation()
    # run_sample_num_40_50()
    




