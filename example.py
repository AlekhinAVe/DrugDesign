import torch
import data as Data
from data.sparse_molecular_dataset import SparseMolecularDataset
import model as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import time
from utils import *
import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')

    # my_code________
    #parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    #parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    #parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')
    parser.add_argument('--log_dir', type=str, default='molgan/logs')
    parser.add_argument('--model_save_dir', type=str, default='molgan/models')
    parser.add_argument('--sample_dir', type=str, default='molgan/samples')
    parser.add_argument('--result_dir', type=str, default='molgan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        val_step = 0
    else:
        wandb_logger = None

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare_data
    data_prepare = SparseMolecularDataset()
    data_prepare.generate('data/gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9)
    data_prepare.save('data/gdb9_9nodes.sparsedataset')

    # data
    data = SparseMolecularDataset()
    data.load(opt['mol_data_dir'])

    # model
    from model.model import DDPM as M
    diffusion = M(opt)
    logger.info('Initial Model Finished')

    # Train
    start_time = time.time()
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    sample_sum = opt['datasets']['val']['data_len']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for i in range(data.data.shape[0] // 16):
                current_step += 1
                if current_step > n_iter:
                    break
                mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
                a = torch.from_numpy(a).to(device).long()  # Adjacency.
                x = torch.from_numpy(x).to(device).long()  # Nodes.
                # converting 'a', 'x' to tensors
                a_tensor = label2onehot(a, opt['b_dim'])
                x_tensor = label2onehot(x, opt['m_dim'])
                diffusion.feed_data(a_tensor, x_tensor)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # print out training information
                if (i + 1) % opt['log_step'] == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, opt['num_iters'])

                    # Log update
                    m0, m1 = all_scores(mols, data, norm=True)  # 'mols' is output of Fake Reward
                    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                    m0.update(m1)
                    for tag, value in m0.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if opt['use_tensorboard']:
                        for tag, value in m0.items():
                            logger.scalar_summary(tag, value, i + 1)
