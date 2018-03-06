from nltk.corpus import stopwords
import pickle
import sys
import string
import torch
import numpy as np
from torch import autograd
from model import Classifier
from torch import nn
vocab=pickle.load(open('embed.pkl','r'))
words=[v[0] for v in vocab]

stop_words = set(stopwords.words('english'))

with open(sys.argv[1],'r') as f:
	lyrics=f.read()

lyrics= lyrics.replace('\n',' ')
lyrics=lyrics.split(' ')
processed_lyrics=[]

for word in lyrics:
	new=word.translate(None,string.punctuation)
	if new not in stop_words:
		processed_lyrics.append(word)
vector=[]
for word in processed_lyrics:
	if word in words:
		vector.append(words.index(word))

vector=np.asarray(vector)

model=torch.load('mymodel.pt')
hidden = model.init_hidden()

vec=autograd.Variable(torch.from_numpy(vector).long())
log_probs,hidden = model(vec,hidden)
maxs, indices = torch.max(log_probs.view(1,-1),1)

if indices.data[0]==0:
	
	print 'label: HAPPY'
else:
	print 'label: SAD'