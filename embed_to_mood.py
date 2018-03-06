import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.autograd import Variable
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
from model import Classifier
# import pdb; pdb.set_trace()

def make_target(label, label_to_index):
	return torch.LongTensor([label_to_index[label]])


data=pickle.load(open('data.pkl','r'))
embeddings=pickle.load(open('embed.pkl','r'))
mat=[]
vocabulary=[]
TRAIN_SIZE = 800

for embedding in embeddings:
	mat.extend(embedding[1])
	vocabulary.append(embedding[0])

mat=np.asarray(mat)
mat.reshape(len(vocabulary),-1)

test_data = data[TRAIN_SIZE:]
data=data[:TRAIN_SIZE]

VOCAB_SIZE = len(vocabulary)
NUM_LABELS = 2
label_to_index = {"happy": 0, "sad": 1}
	





network = Classifier(VOCAB_SIZE,300,50,2)


loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.3)
count=0
in_count = 0
cudnn.benchmark = True
loss_function = loss_function.cuda()
network.cuda()
network.embeddings.weight.data.copy_(torch.from_numpy(mat))
for epoch in range(100):
	for lyrics, label in data:

		hidden = network.init_hidden()
		#import pdb;pdb.set_trace()
		network.zero_grad()
		bag_of_words = Variable(torch.from_numpy(lyrics).long())
		mood = Variable(make_target(label, label_to_index))
		
		bag_of_words.data ,mood.data= bag_of_words.data.cuda(), mood.data.cuda()
		hidden[0].data,hidden[1].data = hidden[0].data.cuda(),hidden[1].data.cuda()

		mood_pred, hidden = network(bag_of_words, hidden)
		
		maxs, indices = torch.max(mood_pred.view(1,-1),1)
		# print indices
		if torch.eq(indices.data, Variable(make_target(label, label_to_index)).data.cuda())[0]==0:
			count=count+1
		loss = loss_function(mood_pred.view(1,-1), mood)
		loss.backward()
		optimizer.step()
		in_count+=1
		# print in_count
	
	# print count
count=0
for lyrics, label in test_data:
	bag_of_words = Variable(torch.from_numpy(lyrics).long())
	
	log_probs = network(bag_of_words)
	maxs, indices = torch.max(log_probs.view(1,-1),1)
	#print count
	if torch.eq(indices.data, Variable(make_target(label, label_to_index)).data)[0]==0:
		count=count+1
# print count
