import yaml
from typing import List


def load_config(config_path: str) -> List[dict]:
    """
    从yaml文件中读取配置

    Args:
        config_path(str): yaml所在目录


    Return:
        config(dict): 配置字典

    """
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config
