import numpy as np
from config import parse_config
from forward import Forward

if __name__ == "__main__":
    cfg = parse_config()


    forward = Forward(cfg)
    forward.read('methane')
    optical_depth = forward.absorption()


