import yaml
import os
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description='Set Pod Count')
    parser.add_argument('--pods',required=False, default=8)
    parser.add_argument('--gpu',required=False, default=1)
    parser.add_argument('--yaml',required=False)
    args=parser.parse_args()
    return args

def main():
	args = parse_args()
	pod_count = args.pods
	gpu_per_pod = args.gpu

	yaml_path = args.yaml #'/mnt/yaml/jobservice_sniper.yaml' #'/home/ubuntu/redkube/nfs/yaml/jobservice_sniper.yaml'

	with open(yaml_path) as f:
		yam = yaml.load(f)

	yam['spec']['parallelism'] = int(pod_count)
	yam['spec']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = int(gpu_per_pod)

	filename = os.path.basename(yaml_path)
	dir_name = os.path.dirname(yaml_path)

	with open(yaml_path,'w') as fw:
		yaml.dump(yam,fw) 


if __name__== "__main__":
	main()
