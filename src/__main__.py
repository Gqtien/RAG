import sys
import yaml

with open(sys.argv[1], "r") as cfg:
    config = yaml.safe_load(cfg)

print(config)
