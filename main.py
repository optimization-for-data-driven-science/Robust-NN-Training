from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict


import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

from model import ConvNet

import numpy as np

# import matplotlib.pyplot as plt

from lossfns import *
from adversary import *
from dataprocess import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
	
	loader_train, loader_test = loadData(args)
	dtype = torch.cuda.FloatTensor
	
	model = unrolled(args, loader_train, loader_test, dtype)

	fname = "model/MNIST_CWM_retain.pth"
	torch.save(model, fname)

	print("Training done, model save to %s :)" % fname)
	
	# fname = "model/CIFAR10_0.03.pth"
	# model = torch.load(fname)

	pgdAttackTest(model, loader_test, dtype)
	fgsmAttackTest(model, loader_test, dtype)


def unrolled(args, loader_train, loader_test, dtype):

	model = ConvNet()
	model = model.type(dtype)
	model.train()
		
	SCHEDULE_EPOCHS = [50, 50] 
	learning_rate = 5e-4
	
	for num_epochs in SCHEDULE_EPOCHS:
		
		print('\nTraining %d epochs with learning rate %.7f' % (num_epochs, learning_rate))
		
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(num_epochs):
			
			print('\nTraining epoch %d / %d ...\n' % (epoch + 1, num_epochs))
			# print(model.training)
			
			for i, (X_, y_) in enumerate(loader_train):

				X = Variable(X_.type(dtype), requires_grad=False)
				y = Variable(y_.type(dtype), requires_grad=False)

				loss = cw_train_unrolled(model, X, y, dtype)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i + 1) % args.print_every == 0:
					print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

					test(model, loader_test, dtype)

			print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))
			
			
		
		learning_rate *= 0.1

	return model


def test(model, loader_test, dtype):
	num_correct = 0
	num_samples = 0
	model.eval()
	for X_, y_ in loader_test:

		X = Variable(X_.type(dtype), requires_grad=False)
		y = Variable(y_.type(dtype), requires_grad=False).long()

		logits = model(X)
		_, preds = logits.max(1)

		num_correct += (preds == y).sum()
		num_samples += preds.size(0)

	accuracy = float(num_correct) / num_samples * 100
	print('\nAccuracy = %.2f%%' % accuracy)
	model.train()

def normal_train(args, loader_train, loader_test, dtype):

	model = ConvNet()
	model = model.type(dtype)
	model.train()
		
	loss_f = nn.CrossEntropyLoss()

	SCHEDULE_EPOCHS = [15] 
	learning_rate = 0.01
	
	for num_epochs in SCHEDULE_EPOCHS:
		
		print('\nTraining %d epochs with learning rate %.4f' % (num_epochs, learning_rate))
		
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(num_epochs):
			
			print('\nTraining epoch %d / %d ...\n' % (epoch + 1, num_epochs))
			# print(model.training)
			
			for i, (X_, y_) in enumerate(loader_train):

				X = Variable(X_.type(dtype), requires_grad=False)
				y = Variable(y_.type(dtype), requires_grad=False).long()

				preds = model(X)

				loss = loss_f(preds, y)
				
				if (i + 1) % args.print_every == 0:
					print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))
			
			test(model, loader_test, dtype)
		
		learning_rate *= 0.1

	return model

def parse_arguments():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='./dataset', type=str,
						help='path to dataset')
	parser.add_argument('--batch-size', default=64, type=int,
						help='size of each batch of cifar-10 training images')
	parser.add_argument('--print-every', default=200, type=int,
						help='number of iterations to wait before printing')

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	main(args)

