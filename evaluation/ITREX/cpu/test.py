import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
args = parser.parse_args()


config = ast.literal_eval(args.config)
print("Parsed config dictionary:", config)
print("="*100)
print(f"{config['model']=}")
