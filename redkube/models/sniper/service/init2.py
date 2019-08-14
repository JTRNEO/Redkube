import sys
import os 
workdir = os.getcwd()
os.chdir('/sniper')
sys.path.insert(0,'lib')
sys.path.insert(0,'SNIPER-mxnet/python')
os.chdir(workdir)
