import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
	"""
	Effective Approaches to Attention-based Neurla machine Translation by Luong et al. 

	Input: de_hidden, en_out
	    -de_hidden: 1, batch, hidden_dimensions
	    -en_out: batch, depth, img_feature_d  
	Output: context, attn
		-context: batch, output_len, dimensions 
		-attn: batch, output_len, dimensions 
	"""
	def __init__(self, method, feature_size, hidden_size):
		super(Attn, self).__init__()


		self.method = method 
		self.hidden_size = hidden_size 

		if self.method == 'general':
			self.attn = nn.Linear(hidden_size, feature_size)
		elif self.method =='concat':
			self.attn = nn.Linear(hidden_size*2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1,hidden_size))

	def forward(self, de_hidden, en_out):

		if de_hidden.size(2) == self.hidden_size:
			de_hidden = self.attn(de_hidden.squeeze(0))  # batch, feature_size

		de_hidden = de_hidden.unsqueeze(2)         #batch, feature_size, 1 
		attn_weight = torch.bmm(en_out, de_hidden)   #batch, depth, 1 
		attn_weight = attn_weight.squeeze(2)
		attn_weight = F.softmax(attn_weight).unsqueeze(2)
		context = torch.bmm(attn_weight.transpose(2,1),en_out) #batch, 1, feature_size

		return context
 

	def score(self, de_hidden, en_out):


		if self.method == 'general':
			score = self.attn(en_out)
			return de_hidden.dot(score)
		elif self.method == 'concat':
			score = self.attn(torch.cat(de_hidden, en_out),1)
			score = self.v.dot(score)
			return score 