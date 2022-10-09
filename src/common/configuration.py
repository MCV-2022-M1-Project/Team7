import os
import yaml
from typing import Any
from munch import DefaultMunch


def load_configuration(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    return DefaultMunch.fromDict(yaml_dict)