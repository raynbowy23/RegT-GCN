import os.path as osp
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal

from utils import unique, IdentityEncoder, preprocess


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


def load_edge_csv(path, mapping=None, src_index_col=None, dst_index_col=None, names=None, encoders=None, edge_cut='neural', **kwargs):
    df = pd.read_csv(path, names=names, **kwargs)

    src = []
    dst = []
    for index in df[src_index_col]:
        try:
            src.append(mapping[index])
        except KeyError:
            mapping[index] = len(mapping)
            src.append(mapping[index])
    for index in df[dst_index_col]:
        try:
            dst.append(mapping[index])
        except:
            mapping[index] = len(mapping)
            dst.append(mapping[index])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    
    if edge_cut == 'random':
        edge_index, edge_mask = random_edge_sampler(edge_index, 0.8)
    elif edge_cut == 'neural':
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    
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
    '''Normal Dataset'''
    def __init__(self, root: str, 
                 transform: Optional[Callable]=None,
                 pre_transform: Optional[Callable]=None,
                 pre_filter: Optional[Callable]=None,
                 train_feature: str='occrate',
                 preprocessed: bool=False,
                 data_size: str='small',
                 edge_cut: str=None):
        self.train_feature = train_feature
        self.root = root
        self.dataset_root = osp.join(self.root, 'data')
        self.processed_root = osp.join(self.root, 'processed', self.train_feature, 'ordinal', '0322')

        self.sc = MinMaxScaler(feature_range=(0,1))
        self.max_list = []
        self.min_list = []
        self.data_size = data_size
        self.preprocessed = preprocessed
        self.edge_cut = edge_cut

        if data_size == "small":
            self.time_range = 14
        elif data_size == "medium":
            self.time_range = 92
        elif data_size == "large":
            self.time_range = 365

        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_file_names(self):
        return osp.join(self.processed_root, 'tpims_data_{}.pkl'.format(self.data_size))

    @property
    def num_nodes(self): 
        return self.data.edge_index.max().item() + 1

    def load_node_data(self, path, index_col, encoders=None):
        data, mapping = load_node_csv(path, index_col, names=['SITE_IDX', 'SITE_ID', 'TIMESTAMP', 'WEEKID', 'DAYID', 'HOURID', 'TRAVEL_TIME', 'TRAVEL_MILE', 'OWNER', 'AMENITY', 'CAPACITY', 'AVAILABLE', 'OCCRATE'], encoders=encoders)
        return data, mapping

    def load_edge_data(self, path, mapping, src_index_col, dst_index_col, encoders=None):
        edge_idx, edge_attr = load_edge_csv(path, mapping, src_index_col, dst_index_col, names=['SRC_IDX', 'SRC', 'DST_IDX', 'DST', 'DIST'], encoders=encoders)
        return edge_idx, edge_attr

    def process(self):
        if self.preprocessed:
            return
        else:
            capacity_stats, amenity_stats, owner_list, mile_marker_list = preprocess(self.dataset_root)
            link_root = osp.join(self.root, 'links', '0322')
            data_dir = osp.join(self.root, 'data')

            link_path = osp.join(link_root, 'link_data.csv')

            node_data_list = []
            sc = MinMaxScaler(feature_range=(0,1))

            # Make mapping for edge data
            dfLOC = pd.read_csv(osp.join(data_dir, 'tpims_location.csv'))
            # Replacement
            # NaN -> 0
            dfLOC = dfLOC[~dfLOC['SITE_ID'].str.startswith(('IL', 'MI', 'MN', 'IN'), na=False)]
            dfLOC = dfLOC.replace({np.nan: 0})

            dfNODE = pd.read_csv(osp.join(data_dir, 'tpims_data_{}.csv'.format(self.data_size)))
            mapping = {index-1: i for i, index in enumerate(range(1, len(dfLOC)))} # site_idx to idx

            # One edge data for all states
            edge_index, edge_attr = self.load_edge_data(link_path, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                        encoders={'DIST': IdentityEncoder(dtype=torch.float)})
            edge_attr = edge_attr.squeeze().clone().detach()
            
            t_prev = datetime.strptime('2022-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
            
            site_id, site_idx = unique(dfLOC['SITE_ID'])
            # available = [0 for i in range(len(dfLOC))]
            site_id_dict = {site: idx for idx, site in enumerate(site_id)}
            available_dict = {site: 0 for site in site_id}
            available = [0 for i in range(len(dfLOC))]


            for i in tqdm(range(6*24*self.time_range)):
                idx = 0

                t = (t_prev + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                t_prev = t_prev.strftime('%Y-%m-%d %H:%M:%S')
                mask = dfNODE['time_stamp'].between(str(t_prev), str(t))
                filtered_df = dfNODE[mask]
                siteId = filtered_df['site_id'].values
                tmp_available = filtered_df['available'].values

                week = int(int(t.split(' ')[0].split('-')[2]) / 7)
                day = datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() # Mon - Sun
                hour = t.split(' ')[1].split(':')[0]
                adj_hour = int(hour)

                ### Creating/preparing csv files first
                site_to_available = dict(zip(siteId, tmp_available))

                # Initialize dataframe
                adj_hour = 0
                temp_s_idx = len(site_id)
                _node_data_list = []
                ## IMPORTANT: All node data must be in the location data.
                for j, site in enumerate(site_id):
                    # Filling up the unknown values
                    if 'IN' not in site[:2] and 'MI' not in site[:2] and 'MN' not in site[:3] and 'IL' not in site[:2]:
                        idx += 1
                        if site in siteId:
                            s_idx = site_id_dict[site]
                            available_value = site_to_available[site]

                            if capacity_stats[j] == 0:
                                capacity_stats[j] = np.finfo(np.float32).eps
                            else:
                                node_dict = {'SITE_IDX': [s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [int(hour)], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [available_value], 'OCCRATE': [available_value/capacity_stats[j]]}
                            available[s_idx] = available_value
                        else:
                            try:
                                # Change available/occrate when the site is not found
                                node_dict = {'SITE_IDX': [temp_s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [adj_hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [int(available[temp_s_idx])], 'OCCRATE': [available[temp_s_idx]/capacity_stats[j]]}
                            except IndexError:
                                node_dict = {'SITE_IDX': [temp_s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [adj_hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [0], 'OCCRATE': [0.0]}
                            temp_s_idx += 1
                                
                        encoders = {'WEEKID': IdentityEncoder(dtype=torch.long), 'DAYID': IdentityEncoder(dtype=torch.long), 'HOURID': IdentityEncoder(dtype=torch.long),
                                'MILE_MARKER': IdentityEncoder(dtype=torch.float), 'OWNER': IdentityEncoder(dtype=torch.long),
                                'AMENITY': IdentityEncoder(dtype=torch.long), 'CAPACITY': IdentityEncoder(dtype=torch.long), self.train_feature.upper(): IdentityEncoder(dtype=torch.float)}

                        if encoders is not None:
                            xs = [encoder(node_dict[col]) for col, encoder in encoders.items()]
                            x = torch.cat(xs, dim=-1)
                            x = torch.nan_to_num(x)

                        _node_data_list.append(x)

                # Upload to csv file
                node_data = torch.cat(_node_data_list, dim=0)
                node_data = torch.from_numpy(sc.fit_transform(node_data))
                node_data_list.append(node_data)

                # Update time
                t_prev = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')

            torch.save((edge_index, edge_attr, node_data_list), osp.join(self.processed_root, 'tpims_data_{}.pkl'.format(self.data_size)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, num_timesteps_in=8, num_timesteps_out=4):

        features = []
        target = []

        edge_index, edge_attr, node_data = torch.load(osp.join(self.processed_root, 'tpims_data_{}.pkl'.format(self.data_size)))
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
                 preprocessed: bool=False,
                 data_size: str='small',
                 decomp_type: str='regional'):
        self.train_feature = train_feature
        self.root = root
        self.dataset_root = osp.join(self.root, 'data')
        self.processed_root = osp.join(self.root, 'processed', self.train_feature, decomp_type, '0322')

        self.sc = MinMaxScaler(feature_range=(0,1))
        self.max_list = []
        self.min_list = []
        self.data_size = data_size
        self.preprocessed = preprocessed
        self.decomp_type = decomp_type

        if data_size == "small":
            self.time_range = 14
        elif data_size == "medium":
            self.time_range = 92
        elif data_size == "large":
            self.time_range = 365

        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_file_names(self):
        return osp.join(self.processed_root, 'tpims_data_{}.pkl'.format(self.data_size))

    @property
    def num_nodes(self): 
        return self.data.edge_index.max().item() + 1

    def load_node_data(self, path, index_col, encoders=None):
        data, mapping = load_node_csv(path, index_col, names=['SITE_IDX', 'SITE_ID', 'TIMESTAMP', 'WEEKID', 'DAYID', 'HOURID', 'TRAVEL_TIME', 'TRAVEL_MILE', 'OWNER', 'AMENITY', 'CAPACITY', 'AVAILABLE', 'OCCRATE'], encoders=encoders)
        return data, mapping

    def load_edge_data(self, path, mapping, src_index_col, dst_index_col, encoders=None):
        edge_idx, edge_attr = load_edge_csv(path, mapping, src_index_col, dst_index_col, names=['SRC_IDX', 'SRC', 'DST_IDX', 'DST', 'DIST'], encoders=encoders)
        return edge_idx, edge_attr

    def process(self):
        if self.preprocessed:
            return
        else:
            capacity_stats, amenity_stats, owner_list, mile_marker_list = preprocess(self.dataset_root)
            link_root = osp.join(self.root, 'links', '0322')
            data_dir = osp.join(self.root, 'data')

            link_path = osp.join(link_root, 'link_data.csv')
            if self.decomp_type == 'regional':
                link_IA = osp.join(link_root, 'link_IA_data.csv')
                link_KS = osp.join(link_root, 'link_KS_data.csv')
                link_KY = osp.join(link_root, 'link_KY_data.csv')
                link_OH = osp.join(link_root, 'link_OH_data.csv')
                link_WI = osp.join(link_root, 'link_WI_data.csv')
            if self.decomp_type == 'random':
                link_IA = osp.join(self.root, 'links/0322/link1_data.csv')
                link_KS = osp.join(self.root, 'links/0322/link2_data.csv')
                link_KY = osp.join(self.root, 'links/0322/link3_data.csv')
                link_OH = osp.join(self.root, 'links/0322/link4_data.csv')
                link_WI = osp.join(self.root, 'links/0322/link5_data.csv')

            node_data_list = []
            sc = MinMaxScaler(feature_range=(0,1))

            # Make mapping for edge data
            dfLOC = pd.read_csv(osp.join(data_dir, 'tpims_location.csv'))
            # Replacement
            # NaN -> 0
            dfLOC = dfLOC[~dfLOC['SITE_ID'].str.startswith(('IL', 'MI', 'MN', 'IN'), na=False)]
            dfLOC = dfLOC.replace({np.nan: 0})

            dfNODE = pd.read_csv(osp.join(data_dir, 'tpims_data_{}.csv'.format(self.data_size)))
            mapping = {index-1: i for i, index in enumerate(range(1, len(dfLOC)))} # site_idx to idx

            # One edge data for all states
            edge_index, edge_attr = self.load_edge_data(link_path, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                        encoders={'DIST': IdentityEncoder(dtype=torch.float)})
            edge_IA_index, edge_IA_attr = self.load_edge_data(link_IA, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                        encoders={'DIST': IdentityEncoder(dtype=torch.float)})
            edge_KS_index, edge_KS_attr = self.load_edge_data(link_KS, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                        encoders={'DIST': IdentityEncoder(dtype=torch.float)})
            edge_KY_index, edge_KY_attr = self.load_edge_data(link_KY, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                        encoders={'DIST': IdentityEncoder(dtype=torch.float)})
            edge_OH_index, edge_OH_attr = self.load_edge_data(link_OH, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                        encoders={'DIST': IdentityEncoder(dtype=torch.float)})
            edge_WI_index, edge_WI_attr = self.load_edge_data(link_WI, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                        encoders={'DIST': IdentityEncoder(dtype=torch.float)})
            edge_attr = edge_attr.squeeze().clone().detach()
            edge_IA_attr = edge_IA_attr.squeeze().clone().detach()
            edge_KS_attr = edge_KS_attr.squeeze().clone().detach()
            edge_KY_attr = edge_KY_attr.squeeze().clone().detach()
            edge_OH_attr = edge_OH_attr.squeeze().clone().detach()
            edge_WI_attr = edge_WI_attr.squeeze().clone().detach()
            
            t_prev = datetime.strptime('2022-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
            
            site_id, site_idx = unique(dfLOC['SITE_ID'])
            # available = [0 for i in range(len(dfLOC))]
            site_id_dict = {site: idx for idx, site in enumerate(site_id)}
            available_dict = {site: 0 for site in site_id}
            available = [0 for i in range(len(dfLOC))]


            for i in tqdm(range(6*24*self.time_range)):
                idx = 0

                t = (t_prev + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                t_prev = t_prev.strftime('%Y-%m-%d %H:%M:%S')
                mask = dfNODE['time_stamp'].between(str(t_prev), str(t))
                filtered_df = dfNODE[mask]
                siteId = filtered_df['site_id'].values
                tmp_available = filtered_df['available'].values

                week = int(int(t.split(' ')[0].split('-')[2]) / 7)
                day = datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() # Mon - Sun
                hour = t.split(' ')[1].split(':')[0]
                adj_hour = int(hour)

                ### Creating/preparing csv files first
                site_to_available = dict(zip(siteId, tmp_available))

                # Initialize dataframe
                adj_hour = 0
                temp_s_idx = len(site_id)
                _node_data_list = []
                ## IMPORTANT: All node data must be in the location data.
                for j, site in enumerate(site_id):
                    # Filling up the unknown values
                    if 'IN' not in site[:2] and 'MI' not in site[:2] and 'MN' not in site[:3] and 'IL' not in site[:2]:
                        idx += 1
                        if site in siteId:
                            s_idx = site_id_dict[site]
                            available_value = site_to_available[site]

                            if capacity_stats[j] == 0:
                                capacity_stats[j] = np.finfo(np.float32).eps
                            else:
                                node_dict = {'SITE_IDX': [s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [int(hour)], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [available_value], 'OCCRATE': [available_value/capacity_stats[j]]}
                            available[s_idx] = available_value
                        else:
                            try:
                                # Change available/occrate when the site is not found
                                node_dict = {'SITE_IDX': [temp_s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [adj_hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [int(available[temp_s_idx])], 'OCCRATE': [available[temp_s_idx]/capacity_stats[j]]}
                            except IndexError:
                                node_dict = {'SITE_IDX': [temp_s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [adj_hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [0], 'OCCRATE': [0.0]}
                            temp_s_idx += 1
                                
                        encoders = {'WEEKID': IdentityEncoder(dtype=torch.long), 'DAYID': IdentityEncoder(dtype=torch.long), 'HOURID': IdentityEncoder(dtype=torch.long),
                                'MILE_MARKER': IdentityEncoder(dtype=torch.float), 'OWNER': IdentityEncoder(dtype=torch.long),
                                'AMENITY': IdentityEncoder(dtype=torch.long), 'CAPACITY': IdentityEncoder(dtype=torch.long), self.train_feature.upper(): IdentityEncoder(dtype=torch.float)}

                        if encoders is not None:
                            xs = [encoder(node_dict[col]) for col, encoder in encoders.items()]
                            x = torch.cat(xs, dim=-1)
                            x = torch.nan_to_num(x)

                        _node_data_list.append(x)

                # Upload to csv file
                node_data = torch.cat(_node_data_list, dim=0)
                node_data = torch.from_numpy(sc.fit_transform(node_data))
                node_data_list.append(node_data)

                # Update time
                t_prev = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')

            torch.save((edge_index, edge_attr, edge_IA_index, edge_IA_attr, edge_KS_index, edge_KS_attr, edge_KY_index,
                        edge_KY_attr, edge_OH_index, edge_OH_attr, edge_WI_index, edge_WI_attr, node_data_list), osp.join(self.processed_root, 'tpims_data_{}.pkl'.format(self.data_size)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx=None, num_timesteps_in=24, num_timesteps_out=12):
        features, target = [], []
        features_IA, target_IA, features_KS, target_KS, features_KY, target_KY, features_OH, target_OH, features_WI, target_WI = [], [], [], [], [], [], [], [], [], []
        edge_index, edge_attr, edge_IA_index, edge_IA_attr, edge_KS_index, edge_KS_attr, edge_KY_index, \
            edge_KY_attr, edge_OH_index, edge_OH_attr, edge_WI_index, edge_WI_attr, node_data = torch.load(osp.join(self.processed_root, 'tpims_data_{}.pkl'.format(self.data_size)))
        node_data = torch.stack(node_data, dim=1).permute(0, 2, 1)
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

        data = StaticGraphTemporalSignal(edge_index, edge_attr, features, target)
        return data, edge_IA_index, edge_KS_index, edge_KY_index, edge_OH_index, edge_WI_index, \
                edge_IA_attr, edge_KS_attr, edge_KY_attr, edge_OH_attr, edge_WI_attr, self.sc, self.max_list, self.min_list
