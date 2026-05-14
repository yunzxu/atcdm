'''
用于读取和写入配置文件的工具模块，便于直接将已有的config文件参数结构化便于调用

'''

import configparser
import os
import yaml
import re


def save_yaml(config, config_path,):
    with open(config_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def load_yaml(config_path):
    with open(config_path, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config


# def load_yaml(config_path):
#     with open(config_path, 'r') as yaml_file:
#         content = yaml_file.read()
#         # 移除 Python 对象标记
#         content = re.sub(r'!!python/object:__main__\.\w+', '', content)
#         config = yaml.safe_load(content)
#     return config

def read_yaml(config_path):
    """Read a YAML configuration file and return the contents as a dictionary."""
    try:
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except FileNotFoundError:
        print(f"Error: The file {config_path} was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


if __name__ == "__main__":
    config_path ='/data/yunzhixu/medical_fusion/hydro_source/train_SVRDM_VQModel_pre_ch48v4_p96_controlt1w_v2_scale1_lrc_from1e-5/config.yml'
    config = load_yaml(config_path)
    print(config)