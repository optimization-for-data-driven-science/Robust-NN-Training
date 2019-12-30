from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

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

import numpy as np

def pgdAttackTest(model, loader_test, dtype):
	
	model.eval()
	epss = [0.0, 0.1, 0.2, 0.3, 0.4]
	MaxIter = 40
	step_size = 1e-2
	
	for eps in epss:

		num_correct = 0
		num_samples = 0
		

		for X_, y_ in loader_test:

			X = Variable(X_.type(dtype), requires_grad=True)
			X_original = Variable(X_.type(dtype), requires_grad=False)
			y = Variable(y_.type(dtype), requires_grad=False).long()

			for i in range(MaxIter):
				logits = model(X)
				loss = F.cross_entropy(logits, y)
				loss.backward()

				with torch.no_grad():
					X.data = X.data + step_size * X.grad.sign() 
					X.data = X_original + (X.data - X_original).clamp(min=-eps, max=eps)
					X.data = X.data.clamp(min=0, max=1)
					X.grad.zero_()

				# if (i % 100 == 0):
					# print("loss =", loss.item())

			# print("loss =", loss.item())

			X.requires_grad = False
			X = (X * 255).long().float() / 255

			logits = model(X)
			_, preds = logits.max(1)

			num_correct += (preds == y).sum()
			num_samples += preds.size(0)

			# print('-' * 20)


		# R = X[0][0].data.cpu().numpy().reshape(32,32)
		# G = X[0][1].data.cpu().numpy().reshape(32,32)
		# B = X[0][2].data.cpu().numpy().reshape(32,32)
		# img = np.dstack((R,G,B))
		# imgplot = plt.imshow(img)
		# plt.show()

		accuracy = float(num_correct) / num_samples * 100
		print('\nAttack using PGD with eps = %.3f, accuracy = %.2f%%' % (eps, accuracy))

def fgsmAttackTest(model, loader_test, dtype):
	
	model.eval()
	epss = [0.0, 0.1, 0.2, 0.3, 0.4]

	for eps in epss:

		num_correct = 0
		num_samples = 0
		

		for X_, y_ in loader_test:

			X = Variable(X_.type(dtype), requires_grad=True)
			y = Variable(y_.type(dtype), requires_grad=False).long()

			logits = model(X)
			loss = F.cross_entropy(logits, y)
			loss.backward()

			with torch.no_grad():
				X += X.grad.sign() * eps
				X.grad.zero_()

			X.requires_grad = False
			X = (X * 255).long().float() / 255
			
			logits = model(X)
			_, preds = logits.max(1)

			num_correct += (preds == y).sum()
			num_samples += preds.size(0)

		# R = X[0][0].data.cpu().numpy().reshape(32,32)
		# G = X[0][1].data.cpu().numpy().reshape(32,32)
		# B = X[0][2].data.cpu().numpy().reshape(32,32)
		# img = np.dstack((R,G,B))
		# imgplot = plt.imshow(img)
		# plt.show()

		accuracy = float(num_correct) / num_samples * 100
		print('\nAttack using FGSM with eps = %.3f, accuracy = %.2f%%' % (eps, accuracy))
