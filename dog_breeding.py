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

    images.append(cv2.resize(cv2.imread("./train/"+name,0), (30, 30)).reshape(1,30,30))
    labels.append(label_dic[os.path.splitext(name)[0]])

images = np.asarray(images)



"""
1-of-k representation
"""

tmp_labels = labels
uniq_labels = set(tmp_labels) # eliminate duplication
num_breeds = len(Counter(labels)) # number of breeds
uniqu_labels_index = dict((label, i) for i, label in enumerate(uniq_labels)) #create dictionary and assign number for each labels

labels_num = [uniqu_labels_index[label] for i,label in enumerate(labels)]
labels_num = np.array(labels_num)

"""
label_one_of = np.zeros((len(labels),len(Counter(labels)) ), dtype=np.integer) #initialization of 1-of-k vector
# create 1-of-k vector
for i, label in enumerate(labels):
    label_one_of[i][uniqu_labels_index[label]] = 1
"""

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


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(2048,1025)
        self.fc2 = nn.Linear(1025,num_breeds)

    def forward(self, x):
        out = self.layer1(x)
        #print out.data.shape
        out = self.layer2(out)
        #print out.data.shape
        out = out.view(out.size(0), -1)
        #print out.data.shape
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out)

    def accuracy(self,output,labels):
        inference =  np.argmax(output.data.numpy(),axis=1)
        answers = labels.data.numpy()
        correction =  np.equal(inference,answers)
        return  np.sum(correction)/float(len(correction))

CNN = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)


"""
Training
"""
batch_size = 1000
learning_rate =0.001
# Data Loader (Input Pipeline)
train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
#print "------------------------"
#print train
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
#print train_loader
#test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
#test_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate)

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        #print images.shape
        images = Variable(images).float()
        labels = Variable(labels).float().type(torch.LongTensor)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = CNN(images)
        #print outputs
        #rint labels
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
        #if i % 100 == 99:    # print every 2000 mini-batches

        accuracy = CNN.accuracy(outputs,labels)
        print
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        print "accuracy :",accuracy
        running_loss = 0.0
        i += 1
print('Finished Training')