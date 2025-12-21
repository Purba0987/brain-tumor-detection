import h5py
import json

with h5py.File('models/model.h5', 'r') as f:
    # Print the model config
    if 'model_config' in f.attrs:
        config = f.attrs['model_config']
        if isinstance(config, bytes):
            config = config.decode('utf-8')
        config = json.loads(config)
        print(json.dumps(config, indent=2))
    else:
        print("No model_config found")