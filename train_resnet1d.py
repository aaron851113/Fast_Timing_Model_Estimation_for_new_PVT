import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import time
import numpy as np 
from sklearn.model_selection import train_test_split
from multi_scale_ori import MSResNet

torch.cuda.set_device(6)

df = pd.read_csv('./train_total.csv')
print(df.head())
print('df.shape :',df.shape)

train , test = train_test_split(df,test_size=0.1)
print('train.shape :',train.shape)
print('test.shape :',test.shape)

value_columns = []
for i in range(1,50):
    string = 'value_'+str(i)
    value_columns.append(string)   

train_y = train[value_columns]
test_y = test[value_columns]

train.pop('ss')
train.pop('tt')
train.pop('ff')
test.pop('ss')
test.pop('tt')
test.pop('ff')
popcol = ['cell_rise','rise_transition','cell_fall','fall_transition','rise_power','fall_power']
#popcol = ['ss','tt','ff','cell_rise','rise_transition','cell_fall','fall_transition','rise_power','fall_power']
trainoh = train[popcol]
testoh = test[popcol]
for col in value_columns + popcol:
    train.pop(col)
    test.pop(col)

    
train_x = train
test_x = test

train_x = train_x.values.astype(np.float32)
train_y = train_y.values.astype(np.float32)
    
test_x = test_x.values.astype(np.float32)
test_y = test_y.values.astype(np.float32)

trainoh = trainoh.values.astype(np.float32)
testoh = testoh.values.astype(np.float32)


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train_x= min_max_scaler.fit_transform(train_x)
test_x= min_max_scaler.fit_transform(test_x)

train_x = np.concatenate((train_x,trainoh), axis=1)
test_x = np.concatenate((test_x,testoh), axis=1)

print('train_x.shape:',train_x.shape)

train_features = torch.from_numpy(train_x).cuda()
train_labels = torch.from_numpy(train_y).cuda()
test_features = torch.from_numpy(test_x).cuda()
test_labels = torch.from_numpy(test_y).cuda()
print('test_labels.shape:',test_labels.shape)
train_set = TensorDataset(train_features,train_labels)
test_set = TensorDataset(test_features,test_labels)

train_data = DataLoader(dataset=train_set,batch_size=1024,shuffle=True)
test_data  = DataLoader(dataset=test_set,batch_size=1024,shuffle=False)

input_channel = train_x.shape[1]
print('input channel(data dimension):',input_channel)

net = MSResNet(input_channel=input_channel,layers=[1, 1, 1, 1], n_output=49).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion =	torch.nn.MSELoss()
losses = []
eval_losses = []

import math
def mse(predict,label):
    total = predict.shape[0]*predict.shape[1]
    mse_total=0.
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if label[i][j] == 0.:
                label[i][j] = 0.000001
            mse_total += min(1,abs(predict[i][j]-label[i][j])/abs(label[i][j])) * min(1,abs(predict[i][j]-label[i][j])/abs(label[i][j]))
    return 100 - 100 * math.sqrt(mse_total/total)

Best = 0.
checkpoint = 0

test_features = test_features.view(test_features.size(0),input_channel,1)

for i in range(3000):
    train_loss = 0
    net.train()
    for tdata,tlabel in train_data:
        tdata = tdata.view(tdata.size(0),input_channel,1)
        y_ = net(tdata)
        loss = criterion(y_, tlabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

    losses.append(train_loss / len(train_data))
    eval_loss = 0
    net.eval()  
    for edata, elabel in test_data:
        edata = edata.view(edata.size(0),input_channel,1)
        y_ = net(edata)
        loss = criterion(y_, elabel)
        eval_loss = eval_loss + loss.item()
    eval_losses.append(eval_loss / len(test_data))
    
    y_pre = net(test_features)
    MSE = mse(y_pre.squeeze().detach().cpu().numpy(),test_labels.squeeze().detach().cpu().numpy())
    if MSE > Best:
        Best = MSE
        checkpoint = i
    print('epoch: {}, trainloss: {:.4f}, evalloss: {:.4f}, evalMSE: {:.4f}, BestMSE: {:.4f}'.format(i, train_loss / len(train_data), eval_loss / len(test_data),MSE,Best))
