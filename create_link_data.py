from os import path as osp
import math
import pandas as pd
import numpy as np
import json
import urllib.request

bingMapsKey = "YOUR BING MAPS LOCATIONS API KEY"

state_name = 'WI'

def getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2):
    '''
    Calculate distance with lat and lon
    Use this codes (Haversine formula): https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
    '''
    R = 6371 # radius of the earth in km
    dLat = deg2rad(lat2-lat1) # deg2rad below

    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c # Distance in km
    return d

def deg2rad(deg):
    return deg * (math.pi/180)

def getDistanceOnMap():
    '''
    Should get actual distance with google map api and it shuold be directed
    '''
    pass

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

    dataset_root = osp.join('dataset', 'data')
    link_root = osp.join('dataset', 'links')
    dfLOC = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_Location0322.csv'))
    print(dfLOC.shape)

    # Replacement
    # NaN -> 0
    dfLOC = dfLOC.replace({np.nan: 0})

    # Convert the number of amenity to integer
    ame_len = [] # amenity number
    for i in range(len(dfLOC['AMENITY'])):
        if dfLOC['AMENITY'][i] != 0:
            ame_len.append(len(dfLOC['AMENITY'][i].replace(' ', '').split(',')))
        else:
            ame_len.append(0)

    # Replace amenity by number
    dfLOC['AMENITY'] = ame_len 

    print(dfLOC['SITE_ID'].unique())
    siteId_uni, siteId_idx = unique(dfLOC['SITE_ID'])
    print(len(siteId_uni))

    # Try with five different seeds and see how RanT-GCN works differently.
    np.random.seed(42)
    print(dfLOC['SITE_ID'].sample(frac=1, random_state=42))
    dfLOC = dfLOC.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
    print(dfLOC)

    for i in range(len(dfLOC['SITE_ID'])):
        src = dfLOC['SITE_ID'][i]
        src_idx = siteId_uni.index(src)

        for j in range(len(dfLOC['SITE_ID'])):
            if src != dfLOC['SITE_ID'][j]:
                dst = dfLOC['SITE_ID'][j]
                # if state_name in src[:2] and state_name in dst[:2]:
                dst_idx = siteId_uni.index(dst)

                lat1 = dfLOC.loc[dfLOC['SITE_ID']==src, 'LATITUDE'].values[0]
                lon1 = dfLOC.loc[dfLOC['SITE_ID']==src, 'LONGITUDE'].values[0]
                lat2 = dfLOC.loc[dfLOC['SITE_ID']==dst, 'LATITUDE'].values[0]
                lon2 = dfLOC.loc[dfLOC['SITE_ID']==dst, 'LONGITUDE'].values[0]
                if ('IN' not in src and 'MI' not in src and 'MIN' not in src) or ('IN' not in dst and 'MI' not in dst and 'MIN' not in dst):
                    if getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) < 40:
                        if (state_name in src) or (state_name in dst):
                            travel_time, travel_mile = distance(lat1, lon1, lat2, lon2)

                            dfNew_link = pd.DataFrame({'SRC_IDX': [src_idx], 'SRC': [src], 'DST_IDX': [dst_idx], 'DST': [dst], 'DIST': [travel_mile]})
                            # Upload to csv file
                            dfNew_link.to_csv(osp.join(link_root, 'link_data.csv'), mode='a', header=False, index=False, encoding='utf-8')

                ## Regional placed IA: 45, KS: 18, KY: 13, OH: 18, WI: 11 and connect every places
                if ('IN' not in src and 'MI' not in src and 'MIN' not in src) or ('IN' not in dst and 'MI' not in dst and 'MIN' not in dst):
                    if getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) < 40:
                        # if (state_name in src) and (state_name in dst):
                        travel_time, travel_mile = distance(lat1, lon1, lat2, lon2)

                        dfNew = pd.DataFrame({'SRC_IDX': [src_idx], 'SRC': [src], 'DST_IDX': [dst_idx], 'DST': [dst], 'DIST': [travel_mile]})
                        # Uploadc, dst)
                        if 'IA' in src and 'IA' in dst:
                            dfNew.to_csv(osp.join(link_root, 'link_IA_data.csv'), mode='a', header=False, index=False, encoding='utf-8')
                        # elif i < 64:
                        elif 'KS' in src and 'KS' in dst:
                            dfNew.to_csv(osp.join(link_root, 'link_KS_data.csv'), mode='a', header=False, index=False, encoding='utf-8')
                        # elif i < 77:
                        elif 'KY' in src and 'KY' in dst:
                            dfNew.to_csv(link_root, 'link_KY_data.csv', mode='a', header=False, index=False, encoding='utf-8')
                        # elif i < 95:
                        elif 'OH' in src and 'OH' in dst:
                            dfNew.to_csv(link_root, 'link_OH_data.csv', mode='a', header=False, index=False, encoding='utf-8')
                        # elif i < 106:
                        elif 'WI' in src and 'WI' in dst:
                            dfNew.to_csv(link_root, 'link_WI_data.csv', mode='a', header=False, index=False, encoding='utf-8')
                else:
                    continue

# site_bound = [46, 64, 77, 95, 106]
# site_num = [[] for _ in range(5)]
            
# for i in range(len(site_bound)):
#     if i > 0:
#         site_num[i] = np.random.randint(site_bound[i] - site_bound[i-1], size=4) + site_bound[i-1]
#     else:
#         site_num[i] = np.random.randint(site_bound[i], size=4)

# sn = 0

# for i in range(0, len(dfLOC['SITE_ID'])):
#     src = dfLOC['SITE_ID'][i]
#     src_idx = siteId_uni.index(src)

#     if sn > 4:
#         sn = 0

#     for j in range(0, len(dfLOC['SITE_ID'])):
#         if src != dfLOC['SITE_ID'][j]:
#             dst = dfLOC['SITE_ID'][j]
#             dst_idx = siteId_uni.index(dst)
#             lat1 = dfLOC.loc[dfLOC['SITE_ID']==src, 'LATITUDE'].values[0]
#             lon1 = dfLOC.loc[dfLOC['SITE_ID']==src, 'LONGITUDE'].values[0]
#             lat2 = dfLOC.loc[dfLOC['SITE_ID']==dst, 'LATITUDE'].values[0]
#             lon2 = dfLOC.loc[dfLOC['SITE_ID']==dst, 'LONGITUDE'].values[0]
            
#             ### Random placed IA: 45, KS: 18, KY: 13, OH: 18, WI: 11 and connect every places
#             if ('IN' not in src and 'MI' not in src and 'MIN' not in src) or ('IN' not in dst and 'MI' not in dst and 'MIN' not in dst):
#                 travel_time, travel_mile = distance(lat1, lon1, lat2, lon2)
#                 # print(src, dst)

#                 dfNew = pd.DataFrame({'SRC_IDX': [src_idx], 'SRC': [src], 'DST_IDX': [dst_idx], 'DST': [dst], 'DIST': [travel_mile]})
#                 # Upload to csv file
#                 # if i < 46 and j < 46:
#                 #     for s in site_num[0]:
#                 #         if i == s or j == s:
#                 #             dfNew.to_csv('dataset/links/0322/link1_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #             sn += 1
#                 # if 46 <= i < 64 and 46 <= j < 64:
#                 #     for s in site_num[1]:
#                 #         if i == s or j == s:
#                 #             dfNew.to_csv('dataset/links/0322/link2_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #             sn += 1
#                 # elif 64 <= i < 77 and 64 <= j < 77:
#                 #     for s in site_num[2]:
#                 #         if i == s or j == s:
#                 #             dfNew.to_csv('dataset/links/0322/link3_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #             sn += 1
#                 # elif 77 <= i < 95 and 77 <= j < 95:
#                 #     for s in site_num[3]:
#                 #         if i == s or j == s:
#                 #             dfNew.to_csv('dataset/links/0322/link4_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #             sn += 1
#                 # elif 95 <= i < 106 and 95 <= j < 106:
#                 #     for s in site_num[4]:
#                 #         if i == s or j == s:
#                 #             dfNew.to_csv('dataset/links/0322/link5_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #             sn += 1
#                 # else:
#                 #     continue

#                 if i < 21 and j < 21:
#                    dfNew.to_csv('dataset/links/0922/link1_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 elif 21 <= i < 42:
#                     if 21 <= j < 42:
#                         dfNew.to_csv('dataset/links/0922/link2_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                     else:
#                         continue
#                 elif 42 <= i < 63:
#                     if 42 <= j < 63:
#                         dfNew.to_csv('dataset/links/0922/link3_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                     else:
#                         continue
#                 elif 63 <= i < 84:
#                     if 63 <= j < 84:
#                         dfNew.to_csv('dataset/links/0922/link4_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                     else:
#                         continue
#                 elif 84 <= i < 106:
#                     if 84 <= j < 106:
#                         dfNew.to_csv('dataset/links/0922/link5_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                     else:
#                         continue
#                 else:
#                     break

#                 ### Random Split
#                 # if i < 37 and j < 37:
#                #     dfNew.to_csv('dataset/links/0322/link1_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 # if 37 <= i < 58:
#                 #     if 37 <= j < 58:
#                 #         dfNew.to_csv('dataset/links/0322/link2_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #     else:
#                 #         continue
#                 # elif 58 <= i < 72:
#                 #     if 58 <= j < 72:
#                 #         dfNew.to_csv('dataset/links/0322/link3_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #     else:
#                 #         continue
#                 # elif 72 <= i < 95:
#                 #     if 72 <= j < 95:
#                 #         dfNew.to_csv('dataset/links/0322/link4_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #     else:
#                 #         continue
#                 # elif 95 <= i < 106:
#                 #     if 95 <= j < 106:
#                 #         dfNew.to_csv('dataset/links/0322/link5_data.csv', mode='a', header=False, index=False, encoding='utf-8')
#                 #     else:
#                 #         continue
#                 # else:
#                 #     break