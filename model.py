from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
class Classifier(nn.Module) :
	def __init__(self,vocab_size,embedding_size,hidden_size,output_size) :
		super(Classifier,self).__init__()
		self.vocab_size=vocab_size
		self.embedding_size=embedding_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		self.embeddings=nn.Embedding(vocab_size, embedding_size)
		
		self.lstm=nn.LSTM(embedding_size,hidden_size)
		self.linearOut = nn.Linear(hidden_size,output_size)
	def forward(self,inputs,hidden) :
		x = self.embeddings(inputs).view(len(inputs),1,-1)
		output,hidden = self.lstm(x,hidden)
		output = output[-1]
		output = self.linearOut(output)
		output = F.log_softmax(output)
		return output,hidden
	
	def init_hidden(self) :
		return (Variable(torch.zeros(1, 1, self.hidden_size)),Variable(torch.zeros(1, 1, self.hidden_size)))
