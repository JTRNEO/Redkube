import time 
import rediswq
import os
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description='inference')
    parser.add_argument('--queue',required=True)
    args=parser.parse_args()
    return args

def judge(args):
	queue_name = args.queue
	q = rediswq.RedisWQ(name=queue_name,host="redis")
  	while not q.empty():
  		time.sleep(1)
  	time.sleep(30)
  	if queue_name == "result":
  		print('Inference Complete')


if __name__ == '__main__':
	args = parse_args()
	judge(args)
