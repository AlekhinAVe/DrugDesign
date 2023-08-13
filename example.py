from data.sparse_molecular_dataset import SparseMolecularDataset
from model.model import SimpleUnet
from torch.optim import Adam
from model.loss import *
from sampling import *
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
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
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--noise_steps', type=int, default=300, help='quantity of steps of noise')
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
    # parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    # parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')
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

    def label2onehot(labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def postprocess(inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare_data
    data_prepare = SparseMolecularDataset()
    data_prepare.generate('data/gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9)
    data_prepare.save('data/gdb9_9nodes.sparsedataset')

    # data
    data = SparseMolecularDataset()
    data.load(opt['mol_data_dir'])

    # parameters
    m_dim = data.atom_num_types
    b_dim = data.bond_num_types

    # model
    model = SimpleUnet()

    # optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train
    start_time = time.time()
    n_iter = opt['train']['n_iter']
    sample_sum = opt['datasets']['val']['data_len']

    # constants
    a_shape = data.data_A[0].shape
    x_shape = data.data_X[0].shape

    # steps and epochs
    current_step = 0
    current_epoch = 0

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
                a_tensor = label2onehot(a, b_dim)
                x_tensor = label2onehot(x, m_dim)

                # adding random level of noise to tensors
                t = torch.randint(0, opt['noise_steps'], (opt['batch_size'],), device=device).long()
                loss = get_loss(model, x_tensor, t, device)
                loss.backward()
                optimizer.step()

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

    # Test
    # Load the trained generator.

    with torch.no_grad():
        # generating
        edges_logits, nodes_logits = sample(device, a_shape, x_shape, )
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), opt['post_method'])

        # Fake Reward
        (edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits), 'hard_gumbel')
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]

        # Log update
        m0, m1 = all_scores(mols, data, norm=True)  # 'mols' is output of Fake Reward
        m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
        m0.update(m1)
        for tag, value in m0.items():
            log += ", {}: {:.4f}".format(tag, value)
