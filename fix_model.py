import h5py
import json

# Open the h5 file and read its model config
with h5py.File("mask_detector_mobilenet.h5", "r+") as f:
    model_config = f.attrs.get("model_config")
    
    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")
    
    config_dict = json.loads(model_config)
    
    # Recursively remove 'groups' key from any DepthwiseConv2D layers
    def remove_groups(layer_config):
        if isinstance(layer_config, dict):
            if layer_config.get('class_name') == 'DepthwiseConv2D':
                layer_config['config'].pop('groups', None)
            for value in layer_config.values():
                remove_groups(value)
        elif isinstance(layer_config, list):
            for item in layer_config:
                remove_groups(item)
    
    remove_groups(config_dict)
    
    # Write the fixed config back
    f.attrs.modify("model_config", json.dumps(config_dict))

print("Model config fixed!")