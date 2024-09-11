import math
from os import path as osp
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import json

import urllib.request

from logging import getLogger, INFO
# logging.basicConfig(filename='./logs/{}.txt'.format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M")))
logger = getLogger(__name__)
logger.setLevel(INFO)

bingMapsKey = "YOUR BING MAPS LOCATIONS API KEY"

def unique(siteId):
    unique_id = []
    unique_idx = []

    for i, site in enumerate(siteId):
        if site not in unique_id:
            unique_id.append(site)
            unique_idx.append(i)

    return unique_id, unique_idx

def distance(lat_ori, long_ori, lat_dest=None, long_dest=None, city_name=None):

    if city_name != None:
        encodedDest = urllib.parse.quote(city_name, safe='')
        routeUrl = "http://dev.virtualearth.net/REST/V1/Routes/Driving?wp.0=" + str(lat_ori) + "," + str(long_ori) + "&wp.1=" + encodedDest + "&key=" + bingMapsKey
    else:
        routeUrl = "http://dev.virtualearth.net/REST/V1/Routes/Driving?wp.0=" + str(lat_ori) + "," + str(long_ori) + "&wp.1=" + str(lat_dest) + "," + str(long_dest) + "&key=" + bingMapsKey

    request = urllib.request.Request(routeUrl)
    response = urllib.request.urlopen(request)

    r = response.read().decode(encoding='utf-8')
    result = json.loads(r)

    travel_time = result["resourceSets"][0]["resources"][0]["routeLegs"][0]["travelDistance"]
    travel_mile = result["resourceSets"][0]["resources"][0]["routeLegs"][0]["travelDuration"]

    return travel_time, travel_mile


if __name__ == '__main__':

    site_dict = {}

    dataset_root = osp.join('dataset', 'data')

    # Load data. Change as you need.
    dfLOC1 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20220301_20220307.csv'))
    dfLOC2 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20220308_20220314.csv'))
    dfLOC3 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20220315_20220321.csv'))   
    dfLOC4 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20220322_20220331.csv'))   
    # Data for generalization
    # dfLOC1 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20210701_20210707.csv'))
    # dfLOC2 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20210708_20210714.csv')) 
    # dfLOC3 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20210715_20210721.csv'))   
    # dfLOC4 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20210722_20210728.csv'))   
    dfLOC = pd.concat([dfLOC1, dfLOC2, dfLOC3, dfLOC4])
    logger.info('Overall shape: {}'.format(dfLOC.shape))
    logger.info('Unique siteId: {}'.format(len(dfLOC['siteId'].unique())))

    # site_id, site_idx = unique(dfLOC['siteId'])
    # for site in site_id:
    #     site_dict[site] = None # initialize dictionary

    dfSTATS = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_Location2.csv')) # Location data
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

    # Get travel_time, travel_mile
    travel_time_list = []
    travel_mile_list = []
    for i in range(len(dfSTATS['LATITUDE'])):
        city_name = dfSTATS['CITY'][i]
        state_name = dfSTATS['STATE'][i]
        lat = dfSTATS['LATITUDE'][i]
        long = dfSTATS['LONGITUDE'][i]
        if city_name != '' or city_name != None or city_name != None:
            travel_time, travel_mile = distance(lat, long, city_name=str(city_name) + ", " + str(state_name))

            travel_time_list.append(travel_time)
            travel_mile_list.append(travel_mile)
        else:
            continue

    # TODO: Traffic Volume

    print(f'Number of site: {len(site_idx)}')


    # Replacement
    # NaN -> 0
    dfLOC = dfLOC.replace({np.nan: 0})

    # For loop time range by 30 mins
    t_prev = datetime.datetime.strptime('2022-03-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
    logger.info('Start analysis time: {}'.format(t_prev))
    available = [0 for i in range(len(dfLOC))]

    for i in tqdm(range(6*24*30)):
        idx = 0

        t = (t_prev + datetime.timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
        t_prev = t_prev.strftime('%Y-%m-%dT%H:%M:%SZ')
        siteId = dfLOC.loc[dfLOC['timestamp'].between(str(t_prev), str(t)), 'siteId'].values
        timestamp = dfLOC.loc[dfLOC['timestamp'].between(str(t_prev), str(t)), 'timestamp'].values
        capacity = dfLOC.loc[dfLOC['timestamp'].between(str(t_prev), str(t)), 'capacity'].values
        tmp_available = dfLOC.loc[dfLOC['timestamp'].between(str(t_prev), str(t)), 'available'].values

        # Get unique siteId within the time range
        siteId_uni, siteId_idx = unique(siteId)

        ### Creating/preparing csv files first
        # Get associated feature values with siteId index
        # Then put info into csv
        t_file = t_prev.replace(':', '-') # change notation for file name

        # Initialize dataframe
        adj_week = 0
        adj_day = 0
        adj_hour = 0
        for j, site in enumerate(site_id):

            # Filling up the unknown values
            if 'IN' not in site and 'MI' not in site and 'MIN' not in site:
                idx += 1
                if site in siteId_uni:
                    s_idx = np.ndarray.tolist(siteId).index(site)
                    ts = datetime.datetime.strptime(timestamp[s_idx], '%Y-%m-%dT%H:%M:%SZ')
                    # week = timestamp[s_idx].split('-')[1]
                    week = int(int(timestamp[s_idx].split('T')[0].split('-')[2]) / 7)
                    day = ts.weekday() # Mon - Sun
                    hour = timestamp[s_idx].split('T')[1].split(':')[0]
                    adj_week = int(week)
                    adj_day = int(day)
                    adj_hour = int(hour)

                    dfNew = pd.DataFrame({'SITE_IDX': [idx], 'SITE_ID': [site], 'TIMESTAMP': [timestamp[s_idx]], 'WEEKID': [week], 'DAYID': [day], 'HOURID': [hour], 'TRAVEL_TIME': [travel_time_list[j]], 'TRAVEL_MILE': [travel_mile_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [tmp_available[s_idx]], 'OCCRATE': [available[s_idx]/capacity_stats[j]]})
                    available[s_idx] = tmp_available[s_idx]
                else:
                    # Change available/occrate when the site is not found
                    dfNew = pd.DataFrame({'SITE_IDX': [idx], 'SITE_ID': [site], 'TIMESTAMP': ['2021-07-01T00:00:00Z'], 'WEEKID': [adj_week], 'DAYID': [adj_day], 'HOURID': [adj_hour], 'TRAVEL_TIME': [travel_time_list[j]], 'TRAVEL_MILE': [travel_mile_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [int(available[s_idx])], 'OCCRATE': [available[s_idx]/capacity_stats[j]]})

                # Upload to csv file
                dfNew.to_csv(osp.join('dataset', 'nodes', '0322', 'node_data_{}.csv'.format(t_file)), mode='a', header=False, index=False, encoding='utf-8')


        # Update time
        t_prev = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')