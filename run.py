import os
import os.path as osp
import datetime
import logging
from logging import getLogger, INFO
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch_geometric_temporal.signal import temporal_signal_split
# from torch.utils.tensorboard import SummaryWriter

from models import *
from load_dataset import *

# Writer will output to ./runs/ directory by default
# writer = SummaryWriter()

# Parser setting
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="seed number")
parser.add_argument("--epochs", default=30, type=int, help="Max epochs")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--decay", default=1e-4, type=float, help="Weight decay")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--bs", "--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--tr", "--train_ratio", default=0.8, type=float, help="Train ratio")
parser.add_argument("--tf", "--train_feature", default="available", type=str, help="Train feature (occrate / avaialble)")
parser.add_argument("--edge_cut", default=None, type=str, help="The type of edge cut (random/neural/None)")
parser.add_argument("--dataset_path", default="./dataset", type=str, help="Dataset path")
parser.add_argument("--checkpoint_path", default="../checkpoints/", type=str, help="Checkpoints path")
parser.add_argument("--dataloading_type", default=2, type=int, help="Dataset number (Truckparking dataset '1' / '2')")
parser.add_argument("--decomp_type", default=None, type=str, help="Regional or Random decomposition type")
parser.add_argument("--num_timesteps_in", default=8, type=int, help="Number of timesteps for input, and large number causes a memory allocation issue")
parser.add_argument("--num_timesteps_out", default=4, type=int, help="Number of timesteps for output, which is normally half of input")
parser.add_argument("--model", default="TemporalGCN", type=str, help="Model name you want to use (TemporalGNN - TGCN SOTA, TemporalGConvLSTM - GConvLSTM, RecurrentGCN - might be stack of simple LSTMs)")
parser.add_argument("--is_preprocessed", action="store_true", help="If the dataset is preprocessed")
parser.add_argument("--is_pretrained", action="store_true")
parser.add_argument("--pretrained_model", default="", type=str, help="Pretrained model name")
parser.add_argument("--pretrained_model_epoch", default="0", type=str, help="Pretrained model epochs")
parser.add_argument("--logs", action="store_true")
args = parser.parse_args()

# Logger setting
if args.logs:
    logging.basicConfig(filename='./logs/{}.txt'.format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M")))
logger = getLogger(__name__)
logger.setLevel(INFO)

# Initial Settings
SEED = args.seed
MAX_EPOCHS = args.epochs
LEARNING_RATE = args.lr
MOMENTUM = args.momentum
WEIGHT_DECAY = args.decay
BATCH_SIZE = args.bs
TRAIN_RATIO = args.tr
TRAIN_FEATURE = args.tf

DATASET_PATH = args.dataset_path
CHECKPOINT_PATH = args.checkpoint_path

pre_transform = None

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Datasets
print("Load Data...")
if args.dataloading_type in [1, 3]:
    # Use parking dataset 1 with StaticGraphTemporalSignal, data stored at './data.pt'
    dataset = TruckParkingDataset1(root=DATASET_PATH, pre_transform=pre_transform, train_feature=TRAIN_FEATURE, edge_cut=args.edge_cut, preprocessed=args.is_preprocessed)
    if args.dataloading_type == 1:
        dataset, edge_index, edge_attr, target_sc, max_list, min_list = dataset.get(num_timesteps_in=args.num_timesteps_in, num_timesteps_out=args.num_timesteps_out)
    elif args.dataloading_type == 3:
        dataset, edge_index, edge_attr, target_sc, max_list, min_list = dataset.custom_get(num_timesteps_in=args.num_timesteps_in, num_timesteps_out=args.num_timesteps_out)
elif args.dataloading_type == 2:
    '''
    For a regional dataset, the number of each state sites are below
    IA: 45, KS: 18, KY: 13, OH: 18, WI: 11
    '''
    # Use parking dataset 2 with StaticGraphTemporalSignal, data stored at './data.pt'. Usually used for RegT-GCN
    dataset = TruckParkingDataset2(root=DATASET_PATH, pre_transform=pre_transform, train_feature=TRAIN_FEATURE, preprocessed=args.is_preprocessed, decomp_type=args.decomp_type)
    dataset, edge_IA_index, edge_KS_index, edge_KY_index, edge_OH_index, edge_WI_index, \
        edge_IA_attr, edge_KS_attr, edge_KY_attr, edge_OH_attr, edge_WI_attr, target_sc, max_list, min_list = dataset.get(num_timesteps_in=args.num_timesteps_in, num_timesteps_out=args.num_timesteps_out)
    edge_IA_index = edge_IA_index.to(device)
    edge_KS_index = edge_KS_index.to(device)
    edge_KY_index = edge_KY_index.to(device)
    edge_OH_index = edge_OH_index.to(device)
    edge_WI_index = edge_WI_index.to(device)
    edge_IA_attr = edge_IA_attr.to(device)
    edge_KS_attr = edge_KS_attr.to(device)
    edge_KY_attr = edge_KY_attr.to(device)
    edge_OH_attr = edge_OH_attr.to(device)
    edge_WI_attr = edge_WI_attr.to(device)

# print(len(set(dataset)))
# print(next(iter(dataset)))
num_nodes = len(next(iter(dataset)).x)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.tr)
# print(len(set(train_dataset)))
# print(len(set(test_dataset)))

if args.model == 'RegionalTemporalGCN' or args.model == 'RandomTemporalGCN':
    model = RegionalTemporalGCN(node_features=8, num_nodes=num_nodes, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device)
elif args.model == 'SpatialGCN':
    model = SpatialGCN(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device)
elif args.model == 'TemporalGCN':
    model = TemporalGCN(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) 
elif args.model == 'TemporalGConvLSTM':
    model = TemporalGConvLSTM(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device)
elif args.model == 'StackedGRU':
    model = StackedGRU(in_channels=args.num_timesteps_in, node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device)
elif args.model == 'ConvStackedTemporalGCN':
    model = ConvStackedTemporalGCN(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) 
elif args.model == 'GraphSAGETemporalGCN':
    model = GraphSAGETemporalGCN(node_features=8, num_nodes=num_nodes, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) 
elif args.model == 'GAT':
    model = GATTemporal(node_features=8, num_nodes=num_nodes, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) 
elif args.model == 'STAEformer': # Not working well
    model = STAEformer(num_nodes=num_nodes, in_steps=args.num_timesteps_in, out_steps=args.num_timesteps_out, tod_embedding_dim=0).to(device)
elif args.model == 'STID':
    model = STID(num_nodes=num_nodes, input_len=args.num_timesteps_in, output_len=args.num_timesteps_out, if_time_in_day=False, if_day_in_week=False).to(device)
elif args.model == 'STNorm':
    model = STNorm(num_nodes=num_nodes, in_dim=8, out_dim=args.num_timesteps_out).to(device)

os.makedirs(osp.join('pretrained', TRAIN_FEATURE, args.model), exist_ok=True)

if args.is_pretrained:
    model.load_state_dict(torch.load(osp.join('pretrained', args.tf, args.model, args.pretrained_model)))
pretrained_idx = args.pretrained_model_epoch

# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print('Net\'s state_dict:')
total_param = 0
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    total_param += np.prod(model.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)
#--------------------------------------------------
print('Optimizer\'s state_dict:')
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])


def l2_loss(w):
    return torch.square(w).sum()


def train():
    loss = 0

    model.train()
    step = 0
    total_loss = 0

    for i, batch in tqdm(enumerate(train_dataset)):

        batch = batch.to(device) 
        
        if args.model == 'StackedGRU':
            out = model(batch.x, batch.edge_index) 
            loss = torch.mean((out[:, -1, :] - batch.y)**2).cpu()
        elif args.model == 'RegionalTemporalGCN' or args.model == 'RandomTemporalGCN':
            out, _ = model(batch.x, batch.edge_index, edge_IA_index, edge_KS_index, edge_KY_index, edge_OH_index, edge_WI_index, \
                                    edge_IA_attr, edge_KS_attr, edge_KY_attr, edge_OH_attr, edge_WI_attr)
            loss = torch.mean((out - batch.y) ** 2).cpu()
        elif args.model == 'STAEformer' or args.model == 'STID' or args.model == 'STNorm':
            # Reshape the input tensor to (batch_size, seq_len, num_nodes, num_features)
            x = batch.x.permute(2, 0, 1).unsqueeze(0)
            out = model(x)

            loss = torch.mean((out - batch.y)**2).cpu()
        else:
            out, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
            loss = torch.mean((out - batch.y)**2).cpu()
        loss.backward()
        total_loss += loss.detach().cpu()
        step += 1
        
    optimizer.step()
    optimizer.zero_grad()

    out_loss = loss.detach()

    return out_loss

# Test phase
@torch.no_grad()
def test():
    mse = []

    model.eval()

    for i, batch in tqdm(enumerate(test_dataset)):
        batch = batch.to(device)
        if args.model == 'StackedGRU':
            out = model(batch.x, batch.edge_index) 
            mse.append(((out[:, -1, :] - batch.y)**2).cpu())
        elif args.model == 'RegionalTemporalGCN' or args.model == 'RandomTemporalGCN':
            out, _ = model(batch.x, batch.edge_index, edge_IA_index, edge_KS_index, edge_KY_index, edge_OH_index, edge_WI_index, \
                                    edge_IA_attr, edge_KS_attr, edge_KY_attr, edge_OH_attr, edge_WI_attr)
            mse.append(((out - batch.y)**2).cpu())
        elif args.model == 'STAEformer' or args.model == 'STID' or args.model == 'STNorm':
            x = batch.x.permute(2, 0, 1).unsqueeze(0)

            out = model(x)
            y = batch.y
            mse.append(((out[0][0] - y)**2).cpu())
        else:
            out, h = model(batch.x, batch.edge_index, batch.edge_attr) 
            mse.append(((out - batch.y)**2).cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt()), float(torch.cat(mse, dim=0).mean())


if __name__ == '__main__':
    for epoch in tqdm(range(MAX_EPOCHS+1)):
        logger.info(f'Epoch: {epoch}')

        train_loss = train()

        rmse, mae = test()
        print("Train Loss: {:.4f}, Test RMSE: {:.4f}, MAE: {:.4f}".format(train_loss, rmse, mae))

        # writer.add_scalar('Loss/train', train_loss, epoch)
        # writer.add_scalar('Accuracy/test', rmse, epoch)

        # Save model
        if epoch % 10 == 0: #and epoch > 0:
            torch.save(model.state_dict(), osp.join('pretrained', TRAIN_FEATURE, args.model, 'model_in{}_out{}_epoch{}.pt'.format(args.num_timesteps_in, args.num_timesteps_out, int(pretrained_idx) + epoch)))

    # writer.close()