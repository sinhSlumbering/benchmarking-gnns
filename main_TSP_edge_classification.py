




"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        






"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.TSP_edge_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device






"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    print(model.parameters())
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import glob
from tqdm import tqdm
import logging

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    """
    Training and validation pipeline for edge classification models.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Record start time
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyperparameters in folder config/
    os.makedirs(os.path.dirname(write_config_file), exist_ok=True)
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""
Dataset: {},
Model: {}

params={}
net_params={}

Total Parameters: {}
""".format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
    
    # Initialize SummaryWriter for TensorBoard
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # Setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    logger.info("Training Graphs: {}".format(len(trainset)))
    logger.info("Validation Graphs: {}".format(len(valset)))
    logger.info("Test Graphs: {}".format(len(testset)))
    logger.info("Number of Classes: {}".format(net_params['n_classes']))

    # Initialize model
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    # Initialize lists for recording metrics
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_f1s, epoch_val_f1s = [], [] 
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        # Import train functions specific for WL-GNNs
        from train.train_TSP_edge_classification import train_epoch_dense as train_epoch
        from train.train_TSP_edge_classification import evaluate_network_dense as evaluate_network
        from functools import partial  # Utility function to pass edge_feat to collate function
        
        train_loader = DataLoader(trainset, shuffle=True, 
                                  collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
        val_loader = DataLoader(valset, shuffle=False, 
                                collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
        test_loader = DataLoader(testset, shuffle=False, 
                                 collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))

    else:
        # Import train functions for all other GCNs
        from train.train_TSP_edge_classification import train_epoch_sparse as train_epoch
        from train.train_TSP_edge_classification import evaluate_network_sparse as evaluate_network

        train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
    # Number of classes
    num_classes = net_params['n_classes']
    
    # Training loop
    try:
        with tqdm(range(params['epochs']), desc='Training') as t:
            for epoch in t:
                start = time.time()
                
                # Training epoch
                if MODEL_NAME in ['RingGNN', '3WLGNN']:  # Different training function for dense GNNs
                    epoch_train_loss, epoch_train_f1, optimizer, train_per_class_correct, train_total = \
                        train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                else:  # For all other models common train function
                    epoch_train_loss, epoch_train_f1, optimizer, train_per_class_correct, train_total = \
                        train_epoch(model, optimizer, device, train_loader, epoch)
                
                # Validation
                epoch_val_loss, epoch_val_f1, val_per_class_correct, val_total = \
                    evaluate_network(model, device, val_loader, epoch)
                # Testing
                epoch_test_loss, epoch_test_f1, test_per_class_correct, test_total = \
                    evaluate_network(model, device, test_loader, epoch)

                # Record metrics
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_f1s.append(epoch_train_f1)
                epoch_val_f1s.append(epoch_val_f1)

                # Log metrics to TensorBoard
                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_f1', epoch_train_f1, epoch)
                writer.add_scalar('val/_f1', epoch_val_f1, epoch)
                writer.add_scalar('test/_f1', epoch_test_f1, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)   

                # Log per-class correct counts
                for cls in range(num_classes):
                    train_cls_correct = train_per_class_correct[cls]
                    val_cls_correct = val_per_class_correct[cls]
                    test_cls_correct = test_per_class_correct[cls]
                    writer.add_scalar(f'train/class_{cls}_correct', train_cls_correct, epoch)
                    writer.add_scalar(f'val/class_{cls}_correct', val_cls_correct, epoch)
                    writer.add_scalar(f'test/class_{cls}_correct', test_cls_correct, epoch)

                # Build status string
                status_str = (
                    f"Epoch {epoch}, Time: {time.time()-start:.2f}s, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                    f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
                    f"Train F1: {epoch_train_f1:.4f}, Val F1: {epoch_val_f1:.4f}, "
                    f"Test F1: {epoch_test_f1:.4f}"
                )
                # Add per-class correct counts to status
                for cls in range(num_classes):
                    status_str += (f", Train_C{cls}_Correct: {train_per_class_correct[cls]}, "
                                   f"Val_C{cls}_Correct: {val_per_class_correct[cls]}, "
                                   f"Test_C{cls}_Correct: {test_per_class_correct[cls]}")
                logger.info(status_str)
                t.set_postfix(epoch=epoch, time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_f1=epoch_train_f1, val_f1=epoch_val_f1,
                              test_f1=epoch_test_f1)

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pkl"))

                # Remove old checkpoints
                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = int(file.split('_')[-1].split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                # Step the scheduler
                scheduler.step(epoch_val_loss)

                # Early stopping
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    logger.info("\n!! LR EQUAL TO MIN LR SET.")
                    break
                    
                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    logger.info('-' * 89)
                    logger.info("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
        
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early because of KeyboardInterrupt')
    
    # Final evaluation on test and train sets
    _, test_f1, test_per_class_correct, test_total = evaluate_network(model, device, test_loader, epoch)
    _, train_f1, train_per_class_correct, train_total = evaluate_network(model, device, train_loader, epoch)
    logger.info("Final Test F1: {:.4f}".format(test_f1))
    logger.info("Final Train F1: {:.4f}".format(train_f1))
    logger.info("Convergence Time (Epochs): {:.4f}".format(epoch))
    logger.info("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    logger.info("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Close TensorBoard writer
    writer.close()

    """
        Write the results in out_dir/results folder
    """
    os.makedirs(os.path.dirname(write_file_name), exist_ok=True)
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""
Dataset: {},
Model: {}

params={}
net_params={}
{}

Total Parameters: {}

FINAL RESULTS
TEST F1: {:.4f}
TRAIN F1: {:.4f}

Per-Class Correct Counts (Test Set):
""".format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
            test_f1, train_f1))
        for cls in range(num_classes):
            f.write("Class {} Correct: {}\n".format(cls, test_per_class_correct[cls]))
        f.write("""
Convergence Time (Epochs): {:.4f}
Total Time Taken: {:.4f}hrs
Average Time Per Epoch: {:.4f}s
""".format(epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))




def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--layer_type', help="Please give a value for layer_type (for GAT and GatedGCN only)")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.layer_type is not None:
        net_params['layer_type'] = layer_type
 

      
    
    # TSP
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].shape[0]
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.concatenate(dataset.train[:][1])))
    net_params['n_classes'] = num_classes
    
    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

    
    
    
    
    
    
    
main()    


























