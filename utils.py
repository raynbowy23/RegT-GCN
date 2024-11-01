import pandas as pd
import os.path as osp
import torch


class IdentityEncoder(object):
    '''Converts a list of floating point values into a PyTorch tensor
    '''
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, _node):
        return torch.tensor(_node).view(-1, 1).to(self.dtype)

def unique(siteId):
    unique_id = []
    unique_idx = []

    for i, site in enumerate(siteId):
        if site not in unique_id:
            unique_id.append(site)
            unique_idx.append(i)

    return unique_id, unique_idx

def preprocess(dataset_root):
    dfSTATS = pd.read_csv(osp.join(dataset_root, 'tpims_location.csv')) # Location data
    site_id, site_idx = unique(dfSTATS['SITE_ID'])
    capacity_stats = dfSTATS['CAPACITY']

    ame_len = [] # amenity number
    if not isinstance(dfSTATS['AMENITY'][0], int):
        for i in range(len(dfSTATS['AMENITY'])):
            if (dfSTATS['AMENITY'][i] != '') or (dfSTATS['AMENITY'][i] != None):
                ame_len.append(len(str(dfSTATS['AMENITY'][i]).replace(' ', '').split(',')))
            else:
                ame_len.append(0)

        # Replace amenity by number
        dfSTATS['AMENITY'] = ame_len 
    amenity_stats = dfSTATS['AMENITY']

    # Ownership
    owner_list = []
    if not isinstance(dfSTATS['OWNERSHIP'][0], int):
        for i in range(len(dfSTATS['OWNERSHIP'])):
            if (dfSTATS['OWNERSHIP'][i] != '') or (dfSTATS['OWNERSHIP'][i] != None):
                if dfSTATS['OWNERSHIP'][i] == 'PU':
                    owner_list.append(0)
                else:
                    owner_list.append(1)
            else:
                owner_list.append(-1)

    # Mile Marker
    mile_marker_list = []
    for i in range(len(dfSTATS['MILE_MARKER'])):
        mile_marker = dfSTATS['MILE_MARKER'][i]
        mile_marker_list.append(mile_marker)

    print(f'Number of site: {len(site_idx)}')

    return capacity_stats, amenity_stats, owner_list, mile_marker_list