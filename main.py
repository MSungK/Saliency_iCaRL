from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import torch
from logging import info
from utils import setup_logger, fix_seed, arg_parse
from os import path as osp

args = arg_parse()
setup_logger(osp.join('Custom_model', args.output))
fix_seed(1234)
numclass=10
feature_extractor=resnet18_cbam()
img_size=32
batch_size=128
task_size=10
memory_size=2000
# epochs=1
epochs=100
learning_rate=2.0


model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate, beta=args.beta, output=args.output)
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))


for i in range(10):
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)