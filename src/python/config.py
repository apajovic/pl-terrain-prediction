# config.py
# Central configuration management
import argparse
import json

class Config:
    def __init__(self, config_path=None, **kwargs):
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        self.config.update(kwargs)

    def __getitem__(self, key):
        # Support dot notation for subkeys, e.g., 'data.train_input_dir'
        if '.' in key:
            keys = key.split('.')
            val = self.config
            for k in keys:
                val = val.get(k, {})
            return val if val != {} else None
        return self.config.get(key)

    def get(self, key, default=None):
        val = self.__getitem__(key)
        return val if val is not None else default

    def as_dict(self):
        return self.config

def get_config(config_path="default_config.json"):
    return Config(config_path)
