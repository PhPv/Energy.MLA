import io
import os
from functools import wraps
from time import time

import requests
import yaml
import collections.abc
import time
import pandas as pd
import pickle
from io import BytesIO
from jinja2 import Template, Undefined


def read_file_check(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fr:
            result = fr.read()
        return result


def fmt_percent(value: float):
    return f"{round(value, 2)} %"


def post_request(api, endpoint, data, headers=None, request_type="POST"):
    headers_dct = {"Content-Type": "application/json"}
    if headers is not None:
        headers_dct.update(headers)
    return requests.request(request_type, f'{api}/{endpoint}', json=data, headers=headers_dct).json()


def get_request(api, endpoint, data, headers=None, request_type="GET"):
    headers_dct = {"Content-Type": "application/json"}
    if headers is not None:
        headers_dct.update(headers)
    return requests.request(request_type, f'{api}/{endpoint}', params=data, headers=headers_dct).json()


def update_nested(d, u):
    for k, v in u.items():
        if k in d and d[k] is None:
            d[k] = {}
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class NullUndefined(Undefined):
    def __getattr__(self, key):
        return ''


def load_yaml_config(pth):
    with open(pth, 'r') as f:
        t = Template(f.read(), undefined=NullUndefined)
        config = yaml.safe_load(t.render())
        config = yaml.safe_load(t.render(config))
    return config


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def mkdirs(paths):
    [mkdir(p) for p in paths]


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def load_obj(name):
    if os.path.exists(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def from_bytes_stream(obj):
    return pickle.load(io.BytesIO(obj))


def to_bytes_stream(obj):
    return io.BytesIO(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap



if __name__ == '__main__':
    import numpy as np
    arr = np.zeros((1, 8292, 10844))




