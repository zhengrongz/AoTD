import os
from omegaconf import OmegaConf

configs = [OmegaConf.load('configs/base_config.yaml')]

# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)