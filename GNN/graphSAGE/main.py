import sys
import os
import torch
import argparse
import pyhocon
import random

from dataCenter import *
from utils import *
from models import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20) # batch 的size
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')  # 默认不使用GCN方法
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./GNN/graphSAGE/experiments.conf')  
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)

	# load data
	ds = args.dataSet
	dataCenter = DataCenter(config)
	dataCenter.load_dataSet(ds)  # 读取数据,如果不填参数就是默认cora数据
	features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)

	# 初始化对象，层数为2， 特征维度是1433，隐藏的embedding层数是128，features，边的连接（还不是矩阵）， device，gcn， aggregate function是mean.
	graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
	graphSage.to(device)
	# 至此，我么此处的 graphSage 是通过 GraphSage 类实例化出来的神经网络模型对象

	num_labels = len(set(getattr(dataCenter, ds+'_labels')))  # label的数量，将属性转化为集合
	classification = Classification(config['setting.hidden_emb_size'], num_labels)
	classification.to(device)
	# 至此，我们此处的 classification 是通过Classification类实例化出来的分类器

	unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), getattr(dataCenter, ds+'_train'), device)
	# 至此，我们此处的unsupervised_loss是通过nsupervisedLoss类实例化的一个损失

	if args.learn_method == 'sup':
		print('GraphSage with Supervised Learning')
	elif args.learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')
	# 例子中，使用的sup，有三种选择

	for epoch in range(args.epochs):
		print('----------------------EPOCH %d-----------------------' % epoch)
		# 返回的是模型
		# 参数是dataCenter, "cora", 两个sage层， 分类器，无监督损失， batch size，normal，deivce, supervise method
		# 进入总的执行语句
		graphSage, classification = apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method)
		if (epoch+1) % 2 == 0 and args.learn_method == 'unsup':
			classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device, args.max_vali_f1, args.name)
		if args.learn_method != 'unsup':
			args.max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, args.max_vali_f1, args.name, epoch)