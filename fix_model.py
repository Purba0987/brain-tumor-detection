import h5py
import json

def fix_dtype_policy(config):
    if isinstance(config, dict):
        if 'dtype' in config and isinstance(config['dtype'], dict) and config['dtype'].get('class_name') == 'DTypePolicy':
            config['dtype'] = config['dtype']['config']['name']
        for key, value in config.items():
            fix_dtype_policy(value)
    elif isinstance(config, list):
        for item in config:
            fix_dtype_policy(item)

with h5py.File('models/model.h5', 'r+') as f:
    if 'model_config' in f.attrs:
        config_str = f.attrs['model_config']
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        config = json.loads(config_str)
        fix_dtype_policy(config)
        new_config_str = json.dumps(config)
        f.attrs['model_config'] = new_config_str.encode('utf-8')
        print("Model config updated successfully")