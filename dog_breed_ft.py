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


def read_labels(file):
  dic = {}
  with open(file) as f:
    reader = f
    for row in reader:
        dic[row.split(",")[0]]  = row.split(",")[1].rstrip() #rstrip(): eliminate "\n"
  return dic

image_names= os.listdir("./train")
label_dic = read_labels("labels.csv")

labels = []
images =[]

for name in image_names[1:]:
    # number of dimensionality should be the same for all images
    images.append(cv2.resize(cv2.imread("./train/"+name,0), (30, 30)).reshape(1,30,30))
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



def accuracy(self):
        for i, (images_val, labels_val) in enumerate(val_loader):

            # print images.shape
            images = Variable(images_val).float()
            labels = Variable(labels_val).float().type(torch.LongTensor)
            outputs = CNN(images)

        inference =  np.argmax(outputs.data.numpy(),axis=1)
        answers = labels.data.numpy()
        correction =  np.equal(inference,answers)
        return  np.sum(correction)/float(len(correction))


class Resnet(nn.Module):
  def __init__(self):
    super(Resnet,self).__init__()
    resnet = models.resnet101(pretrained=True)
    #self.resnet = nn.Sequential(*list(resnet.children())[:-2])
    self.fc = nn.Linear(2048,num_breeds)

  def forward(self,x):
    x = self.resnet(x)
    x = self.fc(x)
    return x

Resnet = Resnet()

"""
Training
"""
batch_size = 1000
learning_rate =0.001
# Data Loader (Input Pipeline)
train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

val = torch.utils.data.TensorDataset(torch.from_numpy(X_validation), torch.from_numpy(Y_validation))
val_loader = torch.utils.data.DataLoader(val, batch_size=len(X_validation), shuffle=True)

#print train_loader
#test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
#test_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Resnet.parameters(), lr=learning_rate)

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        #print images.shape
        images = Variable(images).float()
        labels = Variable(labels).float().type(torch.LongTensor)

        # Forward + Backward + Optimize
        optimizer.zero_grad()

        outputs = Resnet(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        #if i % 100 == 99:    # print every 2000 mini-batches

        accuracy = accuracy()
        print
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        print "accuracy :",accuracy
        running_loss = 0.0
        i += 1
print('Finished Training')

