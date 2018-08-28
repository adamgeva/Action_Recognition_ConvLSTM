import json
from easydict import EasyDict
import os
import datetime


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(params_json, paths_json, new_exp):
    _, config_dict = get_config_from_json(params_json)
    _, paths_dict = get_config_from_json(paths_json)

    config_dict.update(paths_dict)

    config = EasyDict(config_dict)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if new_exp=='Y':
        config.summary_dir = os.path.join("../experiments", now, "summary/")
        config.checkpoint_dir = os.path.join("../experiments", now, "checkpoint/")
    else:
        config.summary_dir = os.path.join(os.path.dirname(params_json), "summary/")
        config.checkpoint_dir = os.path.join(os.path.dirname(params_json), "checkpoint/")
    return config
