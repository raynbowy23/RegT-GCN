import os
import os.path as osp
import numpy as np
import argparse
import datetime
import logging
from logging import getLogger, INFO
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric_temporal.signal import temporal_signal_split


from models import *
from load_dataset import *

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
parser.add_argument("--pretrained_idx", default="30", type=str, help="Pretrained index num")
parser.add_argument("--logs", action="store_true")
parser.add_argument("--visualize", type=bool, default=False, help="Flag for network visualization")
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
    dataset = TruckParkingDataset1(root=DATASET_PATH, pre_transform=pre_transform, train_feature=TRAIN_FEATURE, edge_cut=args.edge_cut, preprocessed=True)
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
    dataset = TruckParkingDataset2(root=DATASET_PATH, pre_transform=pre_transform, train_feature=TRAIN_FEATURE, preprocessed=True, decomp_type=args.decomp_type)
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

if args.model == 'RegionalTemporalGCN':
    model = RegionalTemporalGCN(node_features=8, num_nodes=num_nodes, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) # requires temporal dataset
elif args.model == 'SpatialGCN':
    model = SpatialGCN(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device)
elif args.model == 'RandomTemporalGCN':
    model = RegionalTemporalGCN(node_features=8, num_nodes=num_nodes, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) # requires temporal dataset
elif args.model == 'TemporalGCN':
    model = TemporalGCN(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) # requires temporal dataset
elif args.model == 'TemporalGConvLSTM':
    model = TemporalGConvLSTM(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device)
elif args.model == 'StackedGRU':
    model = StackedGRU(in_channels=args.num_timesteps_in, node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device)
elif args.model == 'ConvStackedTemporalGCN':
    model = ConvStackedTemporalGCN(node_features=8, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) # requires temporal dataset
elif args.model == 'GraphSAGETemporalGCN':
    model = GraphSAGETemporalGCN(node_features=8, num_nodes=num_nodes, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) 
elif args.model == 'GAT':
    model = GATTemporal(node_features=8, num_nodes=num_nodes, periods=args.num_timesteps_in, output_dim=args.num_timesteps_out).to(device) 
elif args.model == 'STAEformer':
    model = STAEformer(num_nodes=num_nodes, in_steps=args.num_timesteps_in, out_steps=args.num_timesteps_out, tod_embedding_dim=0).to(device)
elif args.model == 'STID':
    model = STID(num_nodes=num_nodes, input_len=args.num_timesteps_in, output_len=args.num_timesteps_out, if_day_in_week=False, if_time_in_day=False).to(device)
elif args.model == 'STNorm':
    model = STNorm(num_nodes=num_nodes, in_dim=8, out_dim=args.num_timesteps_out).to(device)


model.load_state_dict(torch.load(osp.join('pretrained', TRAIN_FEATURE, args.model, 'model_in{}_out{}_epoch{}.pt'.format(args.num_timesteps_in, args.num_timesteps_out, int(args.pretrained_idx)))))

# Test phase
@torch.no_grad()
def predict():
    mae = []
    mse = []
    mape = []
    out_list, gt_list, err_list = [], [], []
    cap_max = 50 # You could show the actual usage by changing here.

    model.eval()

    for i, batch in tqdm(enumerate(test_dataset)):

        batch = batch.to(device)

        if args.model == 'StackedGRU':
            out = model(batch.x, batch.edge_index) # No need edge_attr for temporal gnn (will be updated)
            # out = out*cap_max
            out = out[:, -1, :]
            mae.append(np.abs((batch.y - out).cpu()))
            mse.append(((batch.y - out) ** 2).cpu())
            if np.isinf((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95)))).any() == 0:
                mape.append((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95))))
        elif args.model == 'RegionalTemporalGCN' or args.model == 'RandomTemporalGCN':
            out, _ = model(batch.x, batch.edge_index, edge_IA_index, edge_KS_index, edge_KY_index, edge_OH_index, edge_WI_index, \
                                    edge_IA_attr, edge_KS_attr, edge_KY_attr, edge_OH_attr, edge_WI_attr)
            # out = out*cap_max
            # batch.y = batch.y*cap_max
            mae.append(np.abs((batch.y - out).cpu()))
            mse.append(((batch.y - out) ** 2).cpu())
            # Remove inf
            if np.isinf((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95)))).any() == 0:
                mape.append((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95))))
        elif args.model == 'STAEformer' or args.model == 'STID' or args.model == 'STNorm':
            # print(x.shape) # (batch, seq_len, num_nodes, num_features)
            batch.x = batch.x.permute(2, 0, 1).unsqueeze(0)
            out = model(batch.x)
            mae.append(np.abs((batch.y - out).cpu()))
            mse.append(((batch.y - out) ** 2).cpu())
            # Remove inf
            if np.isinf((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95)))).any() == 0:
                mape.append((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95))))
        else:
            out, h = model(batch.x, batch.edge_index, batch.edge_attr) 
            # out = out*cap_max
            # batch.y = batch.y*cap_max
            mae.append(np.abs((batch.y - out).cpu()))
            mse.append(((batch.y - out) ** 2).cpu())
            if np.isinf((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95)))).any() == 0:
                mape.append((np.abs((batch.y - out).cpu() / np.percentile(batch.y.cpu(), q=95))))

        out_list.append(out)
        gt_list.append(batch.y)

    return float(torch.cat(mae, dim=0).mean()), float(torch.cat(mse, dim=0).mean().sqrt()), float(torch.cat(mape, dim=0).mean()) * 100, out_list, gt_list


def visualize_corr(gt_list):
    plt.matshow(np.corrcoef(gt_list.T), 0)


def visualize(out_list, gt_list):
    out_, gt_, gt2_ = [], [], []
    print(len(out_list))
    print(out_list[0].shape)
    print(gt_list[0].shape)

    site_num = 0
    timestep = -1 # out timestep

    if args.model == 'StackedLSTM':
        out2_ = np.asarray([out[site_num][timestep].detach().cpu().numpy() for out in out_list])
        gt2_ = np.asarray([label[site_num][timestep].cpu().numpy() for label in gt_list])
    else:
        out2_ = np.asarray([out[site_num][timestep].detach().cpu().numpy() for out in out_list])
        # out2_ = np.asarray(out_list.cpu())
        gt2_ = np.asarray([label[site_num][timestep].cpu().numpy() for label in gt_list])
    x_ = [i for i in range(len(out2_))]
    if len(x_) < 700:
        # plt.plot(x_[:579], out2_[8:587], color='#0072B2')
        # plt.plot(x_[:579], gt2_[0:579], color='#E69F00')
        ## For time=60
        plt.plot(x_[:242], out2_[8:250], color='#0072B2')
        plt.plot(x_[:242], gt2_[0:242], color='#E69F00')
    else:
        plt.plot(x_[:800], out2_[36:836], color='#0072B2')
        plt.plot(x_[:800], gt2_[0:800], color='#E69F00')
    # plt.plot(x_, out2_, color='blue')
    # plt.plot(x_, gt2_, color='orange')
    plt.show()
    df = pd.DataFrame(out2_[6:806])
    # df = pd.DataFrame(out2_[12:812])
    # df = pd.DataFrame(out2_[36:836])
    # df = pd.DataFrame((gt2_[0:800], out2_[6:806]))
    df.to_csv("result.csv")
    np.set_printoptions(threshold=np.inf)


    ### print values to visualize them externally
    # print(len(gt2_))
    # print(len(out2_))
    # print(repr(gt2_))
    # # print(repr(out2_[3:-1]))
    # # print(repr(gt2_[0:800]))
    # print(repr(out2_[3:-1]))


if __name__ == '__main__':
    mae, rmse, mape, out_list, gt_list = predict()

    # visualize(out_list, gt_list)

    print(f"Test Results: RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")