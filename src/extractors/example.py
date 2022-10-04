import array as np
import numpy as np
from typing import Dict

from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor


@Registry.register_features_extractor
class ExampleExtractor(FeaturesExtractor):
    name: str = "example_extractor"

    def run(features: np.ndarray) -> Dict[str, np.ndarray]:
        #do stuff
        return {
            "output": np.array([1,2,3,4]),
            # Extra results if needed...
        }