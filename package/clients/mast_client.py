import json
import sys
from urllib.parse import quote as urlencode

import numpy as np
import requests
from astropy.table import Table


def mast_query(request):
    # Base API url
    request_url = 'https://mast.stsci.edu/api/v0/invoke'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/" + version}

    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode(req_string)

    # Perform the HTTP request
    resp = requests.post(request_url, data="request=" + req_string, headers=headers)

    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode('utf-8')

    return head, content


def download_request(payload, filename, download_type="file"):
    request_url = 'https://mast.stsci.edu/api/v0.1/Download/' + download_type
    resp = requests.post(request_url, data=payload)

    with open(filename, 'wb') as FLE:
        FLE.write(resp.content)

    return filename


def set_filters(parameters):
    return [{"paramName": p, "values": v} for p, v in parameters.items()]


def set_min_max(min, max):
    return [{'min': min, 'max': max}]


def mast_json2csv(json):
    csv_str = ",".join([x['name'] for x in json['fields']])
    csv_str += "\n"
    csv_str += ",".join([x['type'] for x in json['fields']])
    csv_str += "\n"

    col_names = [x['name'] for x in json['fields']]
    for row in json['data']:
        csv_str += ",".join([str(row.get(col, "nul")) for col in col_names]) + "\n"

    return csv_str


def mast_json2table(json_obj):
    data_table = Table()

    for col, atype in [(x['name'], x['type']) for x in json_obj['fields']]:
        if atype == "string":
            atype = "str"
        if atype == "boolean":
            atype = "bool"
        data_table[col] = np.array([x.get(col, None) for x in json_obj['data']], dtype=atype)

    return data_table


def resolve_name():
    request = {'service': 'Mast.Name.Lookup',
               'params': {'input': 'M101',
                          'format': 'json'},
               }

    headers, out_string = mast_query(request)

    out_data = json.loads(out_string)

    return out_data
