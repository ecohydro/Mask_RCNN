import json
import requests
import wget
import sys
from getpass import getpass
from usgs import api
import pandas as pd

#Form JSON Object to store credentials to generate API Key.
username='rbavery'
password = getpass()

login = r'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/login'

creds={"username": username,
"password": password,
"authType": "",
"catalogId": "EE"}

json_data = {'jsonRequest' : json.dumps(creds)}

r = requests.post(login, data = json_data)
token = r.json()['data'].replace("'",'"') # json needs double quotes

if token == None:
    print("API Key Failed to Generate")

else:
    print("Your API Key Generated Successfully")
    print(token)
    
def check_item_basket(token):
    item_basket_json = {
        "apiKey" : token
    }

    api_string = r'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/'
    get_bd =api_string+"itembasket"

    json_data = {'jsonRequest' : json.dumps(item_basket_json)}

    r = requests.post(get_bd, data = json_data)
    return r.json()

def submit_bulk_order(token):
    request_json = {
        "apiKey" : token
    }

    api_string = r'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/'
    get_bd =api_string+"submitbulkdownloadorder"

    json_data = {'jsonRequest' : json.dumps(request_json)}

    r = requests.post(get_bd, data = json_data)
    return r.json()

#If Key fails to generate please see documentmentation to decode error https://earthexplorer.usgs.gov/inventory/documentation/errors

geojson_aoi = feature_coll = {
"type": "FeatureCollection",
"features": [
{
"type": "Feature",
"properties": {},
"geometry": {
"type": "Polygon",
"coordinates": [
[
[
-104.1064453125,
43.13306116240612
],
[
-104.12841796875,
41.04621681452063
],
[
-102.19482421875,
41.0130657870063
],
[
-102.15087890624999,
40.027614437486655
],
[
-95.07568359375,
40.01078714046552
],
[
-95.625,
40.730608477796636
],
[
-96.13037109375,
42.17968819665961
],
[
-97.27294921875,
43.02071359427862],
[
-98.02001953125,
42.90816007196054
],
[
-98.54736328125,
43.08493742707592
],
[
-104.1064453125,
43.13306116240612
]
]
]
}
}
]
}

def make_date_strings(start_date, end_date):

    return [i.date() for i in pd.date_range(start_date, end_date).tolist() if i.month in [6, 7, 8, 9]]

import numpy as np
start_dates = [str(i) + "-06-01" for i in np.arange(2002, 2020)]
end_dates = [str(i) + "-09-30" for i in np.arange(2002, 2020)]


all_results = []
for start_date, end_date in zip(start_dates, end_dates):
    results = api.search(dataset="ARD_TILE", node="EE", start_date=start_date, end_date=end_date, ll={"longitude":-104.1064453125, "latitude":39.825413103424786}, ur={"longitude":-95.130615234375, "latitude":43.100982876188546}, api_key=token)
    all_results.extend(results['data']['results'])
    print(start_date+" done")

filtered_results = []
for i in all_results:
    if "landsat_etm_c1" in i['browseUrl']:
        pass
    else:
        if i['cloudCover'] == None:
            filtered_results.append(i)
            
        elif i['cloudCover'] <= 10:
            filtered_results.append(i)
        
        else:
            pass

entity_ids = []
for i in filtered_results:
    entity_ids.append("{}".format(i['entityId']))

chunk_size = 50
entity_chunks = [list(i) for i in [*np.array_split(entity_ids, len(entity_ids)/chunk_size)]]
download_lists = []
downloadURL="https://earthexplorer.usgs.gov/inventory/json/v/1.4.1/download"

for chunk in entity_chunks:
    
    orderParms ={'datasetName':"ARD_TILE",
          'products':['SR',
                      'ST',
                      'BT',
                      'QA',
                      'FRB',
                      'METADATA'],
          'entityIds': chunk,
          "apiKey": (token)

          }

    download= requests.get(downloadURL, params={'jsonRequest':json.dumps(orderParms)})

    print(download)
    download_list = download.json()['data']

    download_lists.append(download_list)
    
urlList = []
for lst in download_lists:
    for i in lst:
        urlList.append(i['url'])
        
#Users will need to specify a download directory on their machine.
outDir="/home/ryan/ARD"
mntDir="/mnt/cropmaskperm/tar_downloads"
for URL in urlList:
    print("trying to download "+URL)
    wget.download(URL, out=outDir, bar=None)
    print("downloaded")
print("The Following Scenes have downloaded:")
print(entity_chunks)
print("Download Complete")
