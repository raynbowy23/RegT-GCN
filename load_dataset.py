import os.path as osp
from typing import Callable, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal

from encoders import IdentityEncoder


def load_node_csv(path, idx_col, names, encoders=None, **kwargs):
    '''
    Load node/link csv files
    '''
    df = pd.read_csv(path, index_col=False, names=names, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    
    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping=None, dst_index_col=None, dst_mapping=None, names=None, encoders=None, edge_cut=None, **kwargs):
    df = pd.read_csv(path, names=names, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    batch = torch.tensor([0, 0, 1, 1])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    
    if edge_cut == 'random':
        edge_index, edge_mask = random_edge_sampler(edge_index, 0.8)
    elif edge_cut == 'neural':
        pass
    
    return edge_index, edge_attr


def random_edge_sampler(edge_index, percent, normalization=None):
    '''
    Can be replaced by Randam Temporal GNN.
    percent: The percent of the preserved edges
    '''

    def stub_sampler(normalizatin, cuda):
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index

    if percent >= 1.0:
        return stub_sampler(normalization, edge_index.device)

    row, column = edge_index
    
    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= percent

    edge_index = edge_index[:, edge_mask] 

    return edge_index, edge_mask


class TruckParkingDataset1(Dataset):

    def __init__(self,
                 root: str,
                 edge_window_size: int=10,
                 transform: Optional[Callable]=None,
                 pre_transform: Optional[Callable]=None,
                 pre_filter: Optional[Callable]=None,
                 is_train: bool=True,
                 train_feature: str='occrate',
                 edge_cut: str=None,
                 visualize: bool=False):
        self.edge_window_size = edge_window_size
        self.root = root
        self.is_train = is_train
        self.train_feature = train_feature
        self.processed_root = osp.join(self.root, 'processed', self.train_feature, 'ordinal', '0322')
        self.edge_cut = edge_cut
        self.visualize_adj = visualize

        self.sc = MinMaxScaler(feature_range=(0,1))
        self.max_list = []
        self.min_list = []

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return osp.join(self.processed_root, 'data.pt')

    def num_nodes(self): 
        return self.data.edge_index.max().item() + 1

    def load_node_data(self, path, index_col, encoders=None):
        data, mapping = load_node_csv(path, index_col, names=['SITE_IDX', 'SITE_ID', 'TIMESTAMP', 'WEEKID', 'DAYID', 'HOURID', 'TRAVEL_TIME', 'TRAVEL_MILE', 'OWNER', 'AMENITY', 'CAPACITY', 'AVAILABLE', 'OCCRATE'], encoders=encoders)
        return data, mapping

    def load_edge_data(self, path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
        edge_idx, edge_attr = load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, names=['SRC_IDX', 'SRC', 'DST_IDX', 'DST', 'DIST'], encoders=encoders, edge_cut=self.edge_cut, visualize_adj=self.visualize_adj)
        return edge_idx, edge_attr

    @property
    def raw_file_names(self):
        t_prev = datetime.strptime('2022-03-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
        # t_prev = datetime.strptime('2022-03-15T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
        file_list = []
        _minutes = 10 # time interval every 10 minues
        assert _minutes != 0
        min_per_hour = int(60 / _minutes)

        for i in range(min_per_hour*24*14):
            t = (t_prev + timedelta(minutes=_minutes)).strftime('%Y-%m-%dT%H:%M:%SZ')
            t_prev = t_prev.strftime('%Y-%m-%dT%H:%M:%SZ')

            t_file = t_prev.replace(':', '-') # change notation for file name
            file_name = 'node_data_' + t_file + '.csv'

            # Update time
            t_prev = datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')

            file_list.append(file_name)
        
        return file_list

    def process(self):
    
        node_root = osp.join(self.root, 'nodes/0322')
        link_path = osp.join(self.root, 'links/0322/link_data.csv')
        # processed_root = osp.join(self.root, 'processed', self.train_feature)
        data_store_list = []
        edge_index_list, edge_attr_list, node_data_list, target_list = [], [], [], []

        idx = 0
        for i, node_path in enumerate(self.raw_file_names):
            node_path = osp.join(node_root, node_path)
            if osp.exists(node_path):
                # Read data from 'raw_path'.
                with open(node_path, 'r') as f:
                    data = f.read().split('\n')[:-2]
                    data = [[x for x in line.split(',')] for line in data]

                    stamps = [datetime.strptime(line[2], '%Y-%m-%dT%H:%M:%SZ') for line in data]

                    node_data, mapping = self.load_node_data(node_path, index_col='STATE_IDX', encoders={
                        'WEEKID': IdentityEncoder(dtype=torch.long), 'DAYID': IdentityEncoder(dtype=torch.long), 'HOURID': IdentityEncoder(dtype=torch.long),
                        'TRAVEL_TIME': IdentityEncoder(dtype=torch.float), 'OWNER': IdentityEncoder(dtype=torch.long),
                        'AMENITY': IdentityEncoder(dtype=torch.long), 'CAPACITY': IdentityEncoder(dtype=torch.long), self.train_feature.upper(): IdentityEncoder(dtype=torch.float)})

                edge_index, edge_attr = self.load_edge_data(link_path, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                # num_nodes = edge_index.max().item() + 1

                # edge_attr = [float(line[2]) for line in data] # distance between siteId
                # edge_attr = torch.tensor(edge_attr.squeeze(), dtype=torch.long) # edge_weight
                edge_attr = edge_attr.squeeze().clone().detach()

            node_data = torch.from_numpy(self.sc.fit_transform(node_data))

            max_data = [node_data[i][-1].max() for i in range(node_data.size(0))]
            min_data = [node_data[i][-1].min() for i in range(node_data.size(0))]
            self.max_list.append(max_data)
            self.min_list.append(min_data)


            offset = timedelta(days=3.1)
            graph_idx, factor = [], 1
            # for each 10 minute
            for t in stamps:
                factor = factor if t < stamps[0] + factor * offset else factor + 1
                graph_idx.append(factor - 1)
            graph_idx = torch.tensor(graph_idx, dtype=torch.long)

            data_list = []
            for i in range(graph_idx.max().item() + 1):
                data = Data(x=node_data[:,:-1].type(torch.FloatTensor), edge_index=edge_index.type(torch.LongTensor), edge_attr=edge_attr.type(torch.FloatTensor), 
                            requires_grad_=True)
                data.num_node_features = 8 # capacity
                data.num_nodes = len(mapping)
                data.y = node_data[:,-1].type(torch.FloatTensor) # target == availability (occRate)
                data_list.append(data)

            if self.pre_filter is not None:
                data = self.pre_filter(data)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            node_data_list.append(node_data)

        torch.save((edge_index, edge_attr, node_data_list), osp.join(self.processed_root, 'data.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, num_timesteps_in=8, num_timesteps_out=4):

        features = []
        target = []

        edge_index, edge_attr, node_data = torch.load(osp.join(self.processed_root, 'data.pt'))
        node_data = torch.stack(node_data, dim=1).permute(0,2,1)
        node_data = torch.as_tensor(node_data)
        print(node_data.shape)

        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(node_data.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]
        for i, j in indices:
            features.append((node_data[:,:,i:i+num_timesteps_in]).numpy())
            target.append((node_data[:,-1,i+num_timesteps_in:j]).numpy())

        data = StaticGraphTemporalSignal(edge_index, edge_attr, features, target)

        return data, edge_index, edge_attr, self.sc, self.max_list, self.min_list


class TruckParkingDataset2(Dataset):
    '''Regional/Normal Dataset'''

    def __init__(self, root: str, 
                 transform: Optional[Callable]=None,
                 pre_transform: Optional[Callable]=None,
                 pre_filter: Optional[Callable]=None,
                 train_feature: str='occrate',
                 visualize: bool=False):
        self.train_feature = train_feature
        self.root = root
        self.processed_root = osp.join(self.root, 'processed', self.train_feature, 'regional', '0721')
        self.visualize_adj = visualize

        self.sc = MinMaxScaler(feature_range=(0,1))
        self.max_list = []
        self.min_list = []
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_file_names(self):
        return osp.join(self.processed_root, 'data.pt')

    @property
    def num_nodes(self): 
        return self.data.edge_index.max().item() + 1

    def load_node_data(self, path, index_col, encoders=None):
        data, mapping = load_node_csv(path, index_col, names=['SITE_IDX', 'SITE_ID', 'TIMESTAMP', 'WEEKID', 'DAYID', 'HOURID', 'TRAVEL_TIME', 'TRAVEL_MILE', 'OWNER', 'AMENITY', 'CAPACITY', 'AVAILABLE', 'OCCRATE'], encoders=encoders)
        return data, mapping

    def load_edge_data(self, path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
        edge_idx, edge_attr = load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, names=['SRC_IDX', 'SRC', 'DST_IDX', 'DST', 'DIST'], encoders=encoders, visualize_adj=self.visualize_adj)
        return edge_idx, edge_attr

    @property
    def raw_file_names(self):
        t_prev = datetime.strptime('2021-07-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
        # t_prev = datetime.strptime('2022-09-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
        file_list = []
        _minutes = 10 # time interval every 10 minues
        assert _minutes != 0
        min_per_hour = int(60 / _minutes)

        for i in range(min_per_hour*24*14):
            t = (t_prev + timedelta(minutes=_minutes)).strftime('%Y-%m-%dT%H:%M:%SZ')
            t_prev = t_prev.strftime('%Y-%m-%dT%H:%M:%SZ')

            t_file = t_prev.replace(':', '-') # change notation for file name
            file_name = 'node_data_' + t_file + '.csv'

            # Update time
            t_prev = datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')

            file_list.append(file_name)
        
        return file_list

    def process(self):
    
        node_root = osp.join(self.root, 'nodes/0322')
        # node_root = osp.join(self.root, 'nodes/0622')
        link_path = osp.join(self.root, 'links/0322/link_data.csv')
        link_IA = osp.join(self.root, 'links/0322/link_IA_data.csv')
        link_KS = osp.join(self.root, 'links/0322/link_KS_data.csv')
        link_KY = osp.join(self.root, 'links/0322/link_KY_data.csv')
        link_OH = osp.join(self.root, 'links/0322/link_OH_data.csv')
        link_WI = osp.join(self.root, 'links/0322/link_WI_data.csv')
        # processed_root = osp.join(self.root, 'processed', self.train_feature)
        data_store_list = []
        edge_index_list, edge_attr_list, node_data_list, target_list = [], [], [], []

        idx = 0
        for i, node_path in enumerate(self.raw_file_names):
            node_path = osp.join(node_root, node_path)
            if osp.exists(node_path):
                # Read data from 'raw_path'.
                with open(node_path, 'r') as f:
                    data = f.read().split('\n')[:-2]
                    data = [[x for x in line.split(',')] for line in data]

                    stamps = [datetime.strptime(line[2], '%Y-%m-%dT%H:%M:%SZ') for line in data]

                    node_data, mapping = self.load_node_data(node_path, index_col='STATE_IDX', encoders={
                        'WEEKID': IdentityEncoder(dtype=torch.long), 'DAYID': IdentityEncoder(dtype=torch.long), 'HOURID': IdentityEncoder(dtype=torch.long),
                        'TRAVEL_TIME': IdentityEncoder(dtype=torch.float), 'OWNER': IdentityEncoder(dtype=torch.long),
                        'AMENITY': IdentityEncoder(dtype=torch.long), 'CAPACITY': IdentityEncoder(dtype=torch.long), self.train_feature.upper(): IdentityEncoder(dtype=torch.float)})

                edge_index, edge_attr = self.load_edge_data(link_path, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_IA_index, edge_IA_attr = self.load_edge_data(link_IA, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_KS_index, edge_KS_attr = self.load_edge_data(link_KS, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_KY_index, edge_KY_attr = self.load_edge_data(link_KY, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_OH_index, edge_OH_attr = self.load_edge_data(link_OH, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_WI_index, edge_WI_attr = self.load_edge_data(link_WI, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_attr = edge_attr.squeeze().clone().detach()
                edge_IA_attr = edge_IA_attr.squeeze().clone().detach()
                edge_KS_attr = edge_KS_attr.squeeze().clone().detach()
                edge_KY_attr = edge_KY_attr.squeeze().clone().detach()
                edge_OH_attr = edge_OH_attr.squeeze().clone().detach()
                edge_WI_attr = edge_WI_attr.squeeze().clone().detach()

            # Normalize
            node_data = torch.from_numpy(self.sc.fit_transform(node_data))

            # Get max and min available for every timestep
            max_data = [node_data[i][-1].max() for i in range(node_data.size(0))]
            min_data = [node_data[i][-1].min() for i in range(node_data.size(0))]
            self.max_list.append(max_data)
            self.min_list.append(min_data)

            offset = timedelta(days=3.1)
            graph_idx, factor = [], 1
            # for each 10 minute
            for t in stamps:
                factor = factor if t < stamps[0] + factor * offset else factor + 1
                graph_idx.append(factor - 1)
            graph_idx = torch.tensor(graph_idx, dtype=torch.long)

            data_list = []
            for i in range(graph_idx.max().item() + 1):
                data = Data(x=node_data[:,:-1].type(torch.FloatTensor), edge_index=edge_index.type(torch.LongTensor), edge_attr=edge_attr.type(torch.FloatTensor), 
                            edge_IA_index=edge_IA_index.type(torch.LongTensor), edge_IA_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_KS_index=edge_KS_index.type(torch.LongTensor), edge_KS_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_KY_index=edge_KY_index.type(torch.LongTensor), edge_KY_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_OH_index=edge_OH_index.type(torch.LongTensor), edge_OH_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_WI_index=edge_WI_index.type(torch.LongTensor), edge_WI_attr=edge_IA_attr.type(torch.FloatTensor),
                            requires_grad_=True)
                data.num_node_features = 8 # capacity
                data.num_nodes = len(mapping)
                data.y = node_data[:,-1].type(torch.FloatTensor)# target == availability (occRate)
                data_list.append(data)


            if self.pre_filter is not None:
                data = self.pre_filter(data)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)


            node_data_list.append(node_data)


        torch.save((edge_index, edge_attr, edge_IA_index, edge_IA_attr, edge_KS_index, edge_KS_attr, edge_KY_index,
                    edge_KY_attr, edge_OH_index, edge_OH_attr, edge_WI_index, edge_WI_attr, node_data_list), osp.join(self.processed_root, 'data.pt'))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx=None, num_timesteps_in=24, num_timesteps_out=12):
        features, target = [], []
        features_IA, target_IA, features_KS, target_KS, features_KY, target_KY, features_OH, target_OH, features_WI, target_WI = [], [], [], [], [], [], [], [], [], []
        edge_index, edge_attr, edge_IA_index, edge_IA_attr, edge_KS_index, edge_KS_attr, edge_KY_index, \
        edge_KY_attr, edge_OH_index, edge_OH_attr, edge_WI_index, edge_WI_attr, node_data = torch.load(osp.join(self.processed_root, 'data.pt'))
        node_data = torch.stack(node_data, dim=1).permute(0,2,1)
        node_data = torch.as_tensor(node_data)
        print(node_data.shape)

        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(node_data.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]
        for i, j in indices:
            features.append((node_data[:,:,i:i+num_timesteps_in]).numpy())
            target.append((node_data[:,-1,i+num_timesteps_in:j]).numpy())
            features_IA.append((node_data[:45,:,i:i+num_timesteps_in]).numpy())
            target_IA.append((node_data[:45,-1,i+num_timesteps_in:j]).numpy())
            features_KS.append((node_data[45:63,:,i:i+num_timesteps_in]).numpy())
            target_KS.append((node_data[45:63,-1,i+num_timesteps_in:j]).numpy())
            features_KY.append((node_data[63:76,:,i:i+num_timesteps_in]).numpy())
            target_KY.append((node_data[63:76,-1,i+num_timesteps_in:j]).numpy())
            features_OH.append((node_data[76:94,:,i:i+num_timesteps_in]).numpy())
            target_OH.append((node_data[76:94,-1,i+num_timesteps_in:j]).numpy())
            features_WI.append((node_data[94:105,:,i:i+num_timesteps_in]).numpy())
            target_WI.append((node_data[94:105,-1,i+num_timesteps_in:j]).numpy())
        mean = sum(target) / len(target)
        variance = sum([((x - mean) ** 2) for x in target]) / len(target)
        std = variance ** 0.5
        data = StaticGraphTemporalSignal(edge_index, edge_attr, features, target)
        return data, edge_IA_index, edge_KS_index, edge_KY_index, edge_OH_index, edge_WI_index, \
                edge_IA_attr, edge_KS_attr, edge_KY_attr, edge_OH_attr, edge_WI_attr, self.sc, self.max_list, self.min_list


class TruckParkingDataset3(Dataset):
    '''Randomized Truck Parking Dataset'''

    def __init__(self, root: str, 
                 transform: Optional[Callable]=None,
                 pre_transform: Optional[Callable]=None,
                 pre_filter: Optional[Callable]=None,
                 train_feature: str='occrate',
                 visualize: bool=False):
        self.train_feature = train_feature
        self.root = root
        self.processed_root = osp.join(self.root, 'processed', self.train_feature, 'random', '0322')
        self.visualize_adj = visualize

        self.sc = MinMaxScaler(feature_range=(0,1))
        self.max_list = []
        self.min_list = []
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_file_names(self):
        return osp.join(self.processed_root, 'data.pt')

    @property
    def num_nodes(self): 
        return self.data.edge_index.max().item() + 1

    def load_node_data(self, path, index_col, encoders=None):
        data, mapping = load_node_csv(path, index_col, names=['SITE_IDX', 'SITE_ID', 'TIMESTAMP', 'WEEKID', 'DAYID', 'HOURID', 'TRAVEL_TIME', 'TRAVEL_MILE', 'OWNER', 'AMENITY', 'CAPACITY', 'AVAILABLE', 'OCCRATE'], encoders=encoders)
        return data, mapping

    def load_edge_data(self, path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
        edge_idx, edge_attr = load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, names=['SRC_IDX', 'SRC', 'DST_IDX', 'DST', 'DIST'], encoders=encoders, visualize_adj=self.visualize_adj)
        return edge_idx, edge_attr

    @property
    def raw_file_names(self):
        t_prev = datetime.strptime('2022-03-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
        # t_prev = datetime.strptime('2022-03-15T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
        
        file_list = []
        _minutes = 10 # time interval every 10 minues
        assert _minutes != 0
        min_per_hour = int(60 / _minutes)

        for i in range(min_per_hour*24*14):
            t = (t_prev + timedelta(minutes=_minutes)).strftime('%Y-%m-%dT%H:%M:%SZ')
            t_prev = t_prev.strftime('%Y-%m-%dT%H:%M:%SZ')

            t_file = t_prev.replace(':', '-') # change notation for file name
            file_name = 'node_data_' + t_file + '.csv'

            # Update time
            t_prev = datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')

            file_list.append(file_name)
        
        return file_list

    def process(self):
    
        node_root = osp.join(self.root, 'nodes/0322')
        link_path = osp.join(self.root, 'links/0322/link_data.csv')
        link_IA = osp.join(self.root, 'links/0322/link1_data.csv')
        link_KS = osp.join(self.root, 'links/0322/link2_data.csv')
        link_KY = osp.join(self.root, 'links/0322/link3_data.csv')
        link_OH = osp.join(self.root, 'links/0322/link4_data.csv')
        link_WI = osp.join(self.root, 'links/0322/link5_data.csv')
        # processed_root = osp.join(self.root, 'processed', self.train_feature)
        data_store_list = []
        edge_index_list, edge_attr_list, node_data_list, target_list = [], [], [], []

        idx = 0
        for i, node_path in enumerate(self.raw_file_names):
            node_path = osp.join(node_root, node_path)
            if osp.exists(node_path):
                # Read data from 'raw_path'.
                with open(node_path, 'r') as f:
                    data = f.read().split('\n')[:-2]
                    data = [[x for x in line.split(',')] for line in data]

                    stamps = [datetime.strptime(line[2], '%Y-%m-%dT%H:%M:%SZ') for line in data]

                    node_data, mapping = self.load_node_data(node_path, index_col='STATE_IDX', encoders={
                        'WEEKID': IdentityEncoder(dtype=torch.long), 'DAYID': IdentityEncoder(dtype=torch.long), 'HOURID': IdentityEncoder(dtype=torch.long),
                        'TRAVEL_TIME': IdentityEncoder(dtype=torch.float), 'OWNER': IdentityEncoder(dtype=torch.long),
                        'AMENITY': IdentityEncoder(dtype=torch.long), 'CAPACITY': IdentityEncoder(dtype=torch.long), self.train_feature.upper(): IdentityEncoder(dtype=torch.float)})

                edge_index, edge_attr = self.load_edge_data(link_path, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_IA_index, edge_IA_attr = self.load_edge_data(link_IA, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_KS_index, edge_KS_attr = self.load_edge_data(link_KS, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_KY_index, edge_KY_attr = self.load_edge_data(link_KY, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_OH_index, edge_OH_attr = self.load_edge_data(link_OH, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_WI_index, edge_WI_attr = self.load_edge_data(link_WI, src_index_col='SRC_IDX', src_mapping=mapping, dst_index_col='DST_IDX',
                                                            dst_mapping=mapping, encoders={'DIST': IdentityEncoder(dtype=torch.float)})
                edge_attr = edge_attr.squeeze().clone().detach()
                edge_IA_attr = edge_IA_attr.squeeze().clone().detach()
                edge_KS_attr = edge_KS_attr.squeeze().clone().detach()
                edge_KY_attr = edge_KY_attr.squeeze().clone().detach()
                edge_OH_attr = edge_OH_attr.squeeze().clone().detach()
                edge_WI_attr = edge_WI_attr.squeeze().clone().detach()

            # Normalize
            node_data = torch.from_numpy(self.sc.fit_transform(node_data))

            # Get max and min available for every timestep
            max_data = [node_data[i][-1].max() for i in range(node_data.size(0))]
            min_data = [node_data[i][-1].min() for i in range(node_data.size(0))]
            self.max_list.append(max_data)
            self.min_list.append(min_data)

            offset = timedelta(days=3.1)
            graph_idx, factor = [], 1
            # for each 10 minute
            for t in stamps:
                factor = factor if t < stamps[0] + factor * offset else factor + 1
                graph_idx.append(factor - 1)
            graph_idx = torch.tensor(graph_idx, dtype=torch.long)

            data_list = []
            for i in range(graph_idx.max().item() + 1):
                data = Data(x=node_data[:,:-1].type(torch.FloatTensor), edge_index=edge_index.type(torch.LongTensor), edge_attr=edge_attr.type(torch.FloatTensor), 
                            edge_IA_index=edge_IA_index.type(torch.LongTensor), edge_IA_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_KS_index=edge_KS_index.type(torch.LongTensor), edge_KS_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_KY_index=edge_KY_index.type(torch.LongTensor), edge_KY_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_OH_index=edge_OH_index.type(torch.LongTensor), edge_OH_attr=edge_IA_attr.type(torch.FloatTensor),
                            edge_WI_index=edge_WI_index.type(torch.LongTensor), edge_WI_attr=edge_IA_attr.type(torch.FloatTensor),
                            requires_grad_=True)
                data.num_node_features = 8 # capacity
                data.num_nodes = len(mapping)
                data.y = node_data[:,-1].type(torch.FloatTensor)# target == availability (occRate)
                data_list.append(data)

            if self.pre_filter is not None:
                data = self.pre_filter(data)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            node_data_list.append(node_data)

        torch.save((edge_index, edge_attr, edge_IA_index, edge_IA_attr, edge_KS_index, edge_KS_attr, edge_KY_index,
                    edge_KY_attr, edge_OH_index, edge_OH_attr, edge_WI_index, edge_WI_attr, node_data_list), osp.join(self.processed_root, 'data.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx=None, num_timesteps_in=24, num_timesteps_out=12):
        features, target = [], []
        features_IA, target_IA, features_KS, target_KS, features_KY, target_KY, features_OH, target_OH, features_WI, target_WI = [], [], [], [], [], [], [], [], [], []
        edge_index, edge_attr, edge_IA_index, edge_IA_attr, edge_KS_index, edge_KS_attr, edge_KY_index, \
        edge_KY_attr, edge_OH_index, edge_OH_attr, edge_WI_index, edge_WI_attr, node_data = torch.load(osp.join(self.processed_root, 'data.pt'))
        node_data = torch.stack(node_data, dim=1).permute(0,2,1)
        node_data = torch.as_tensor(node_data)
        print(node_data.shape)

        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(node_data.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]
        for i, j in indices:
            features.append((node_data[:,:,i:i+num_timesteps_in]).numpy())
            target.append((node_data[:,-1,i+num_timesteps_in:j]).numpy())
            features_IA.append((node_data[:45,:,i:i+num_timesteps_in]).numpy())
            target_IA.append((node_data[:45,-1,i+num_timesteps_in:j]).numpy())
            features_KS.append((node_data[45:63,:,i:i+num_timesteps_in]).numpy())
            target_KS.append((node_data[45:63,-1,i+num_timesteps_in:j]).numpy())
            features_KY.append((node_data[63:76,:,i:i+num_timesteps_in]).numpy())
            target_KY.append((node_data[63:76,-1,i+num_timesteps_in:j]).numpy())
            features_OH.append((node_data[76:94,:,i:i+num_timesteps_in]).numpy())
            target_OH.append((node_data[76:94,-1,i+num_timesteps_in:j]).numpy())
            features_WI.append((node_data[94:105,:,i:i+num_timesteps_in]).numpy())
            target_WI.append((node_data[94:105,-1,i+num_timesteps_in:j]).numpy())
        mean = sum(target) / len(target)
        variance = sum([((x - mean) ** 2) for x in target]) / len(target)
        std = variance ** 0.5
        data = StaticGraphTemporalSignal(edge_index, edge_attr, features, target)
        return data, edge_IA_index, edge_KS_index, edge_KY_index, edge_OH_index, edge_WI_index, \
                edge_IA_attr, edge_KS_attr, edge_KY_attr, edge_OH_attr, edge_WI_attr, self.sc, self.max_list, self.min_list