from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import os
import cv2
import torch.optim as optim
import torch.utils.data
from torchvision import models

print "GPU",torch.cuda.is_available()
def read_labels(file):
  dic = {}
  with open(file) as f:
    reader = f
    for row in reader:
        dic[row.split(",")[0]]  = row.split(",")[1].rstrip() #rstrip(): eliminate "\n"
  return dic

image_names= os.listdir("../train")
label_dic = read_labels("../labels.csv")

labels = []
images =[]

for name in image_names[1:]:
    # number of dimensionality should be the same for all images
    images.append(cv2.resize(cv2.imread("../train/"+name), (60, 60)).reshape(3,60,60))
    labels.append(label_dic[os.path.splitext(name)[0]])

images = np.asarray(images)



"""
Assign numbers for each labels
"""

tmp_labels = labels
uniq_labels = set(tmp_labels) # eliminate duplication
num_breeds = len(Counter(labels)) # number of breeds
uniqu_labels_index = dict((label, i) for i, label in enumerate(uniq_labels)) #create dictionary and assign number for each labels

labels_num = [uniqu_labels_index[label] for i,label in enumerate(labels)]
labels_num = np.array(labels_num)


"""
Data distribution
"""
N = len(images)
N_train = int(N * 0.7)
N_test = int(N*0.2)

#label_one_of = np.array(labels).astype('int')
#print "hhh",type(images)
X_train, X_tmp, Y_train, Y_tmp = train_test_split(images, labels_num, train_size=N_train)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_tmp, Y_tmp, test_size=N_test)

"""
Model Definition
"""



def accuracy():
	correct = 0
	total = 0
	for images_val, labels_val in test_loader:
    		images_val = Variable(images_val.cuda()).float()
        	outputs_val = model_ft(images_val)
        #outputs = avgpool(outputs)
        	outputs_val = outputs_val.view(outputs_val.size(0), -1)
        	outputs_val = fc(outputs_val)    		
		_, predicted = torch.max(outputs_val.data, 1)
    		total += labels_val.size(0)
    		correct += (predicted.cpu() == labels_val).sum() 

        print "accuracy :",float(correct)/total 
"""  
      for i, (images_val, labels_val) in enumerate(val_loader):

            # print images.shape
            images_val = Variable(images_val.cuda()).float()
            labels_val = Variable(labels_val.cuda()).float().type(torch.cuda.LongTensor)
            outputs_val = model_ft(images_val)
	    outputs_val = outputs_val.view(outputs_val.size(0), -1)
            outputs_val = fc(outputs_val)
	    _, predicted = torch.max(outputs_val.data, 1)	    
	    total += labels_val.size(0)
            correct += (predicted == labels_val).sum()

"""	


        #k\inference =  np.argmax(outputs_val,axis=1)
        #values, indices = torch.max(outputs_val, 1)
        #correction =  torch.equal(indices,labels_val)
        #return  np.sum(correction)/float(len(correction))

model_ft=models.resnet18(pretrained=True)
model_ft = nn.Sequential(*list(model_ft.children())[:-2])
#num_ftrs = model_ft.fc.in_features
avgpool = nn.AvgPool2d(7,padding = 1)
fc = nn.Linear(2048,num_breeds)

model_ft = model_ft.cuda()
avgpool = avgpool.cuda()
fc = fc.cuda()

"""
Training
"""
batch_size = 500
learning_rate =0.001
# Data Loader (Input Pipeline)
train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

#print train_loader
#test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
#test_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=learning_rate)

for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        #print images.shape
        images = Variable(images.cuda()).float()
        labels = Variable(labels.cuda()).float().type(torch.cuda.LongTensor)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
	#print images.data.numpy().size
        outputs = model_ft(images)
        #outputs = avgpool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
	outputs = fc(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        #if i % 100 == 99:    # print every 2000 mini-batches

    print
    print "epochs :",epoch
    print "loss :", float(running_loss) / 2000
    accuracy()

print('Finished Training')
