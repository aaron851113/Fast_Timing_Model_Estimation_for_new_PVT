import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import time
import numpy as np
import math
from sklearn.model_selection import train_test_split
#from multi_scale_renet import MSResNet
#from multi_scale_ori import MSResNet
from SENet import MSResNet

torch.cuda.set_device(7)
class mseLoss(torch.nn.Module):
    def __init__(self):
        super( mseLoss, self).__init__()
    def forward(self, output, timing):
        batch, wh = output.shape
        timing[timing == 0.] = 1e-5
        diff = torch.abs(timing - output)
        diff = diff / torch.abs(timing)
        diff[diff >= 1] = 1
        diff = torch.pow(diff, 2)
        diff = torch.sum(diff.view(diff.shape[0], -1), dim = 1)
        diff = diff / wh
        diff = torch.sqrt(diff)
        diff = torch.sum(diff) / batch
        return diff


def mse(predict,label):
    total = predict.shape[0]*predict.shape[1]
    mse_total=0.
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if label[i][j] == 0.:
                label[i][j] = 1e-5
            mse_total += min(1,abs(predict[i][j]-label[i][j])/abs(label[i][j])) * min(1,abs(predict[i][j]-label[i][j])/abs(label[i][j]))
    return 100 - 100 * math.sqrt(mse_total/total)

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

df = pd.read_csv('./train_total.csv')
print(df.head())
print('df.shape :',df.shape)

train , test = train_test_split(df,test_size=0.1)


value_columns = []
for i in range(1,50):
    string = 'value_'+str(i)
    value_columns.append(string)   

train_y = train[value_columns]
test_y = test[value_columns]

#popcol = ['cell_rise','rise_transition','cell_fall','fall_transition','rise_power','fall_power']
popcol = ['cell_rise','rise_transition','cell_fall','fall_transition','rise_power','fall_power']

trainoh = train[popcol]
testoh = test[popcol]
for col in value_columns + popcol + ['ss','tt','ff',]:
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
#train_x=preprocessing.normalize(train_x, norm='l2')
#test_x=preprocessing.normalize(test_x, norm='l2')

train_x = np.concatenate((train_x,trainoh), axis=1)
test_x = np.concatenate((test_x,testoh), axis=1)


train_features = torch.from_numpy(train_x).cuda()
train_labels = torch.from_numpy(train_y).cuda()
test_features = torch.from_numpy(test_x).cuda()
test_labels = torch.from_numpy(test_y).cuda()
train_set = TensorDataset(train_features,train_labels)
test_set = TensorDataset(test_features,test_labels)

print('train.shape :',train_features.shape)
print('test.shape :',test_features.shape)


input_channel = 25
batch_size = 1024
Epoch = 300
base_lr = 0.001
train_data = DataLoader(dataset=train_set,batch_size=batch_size ,shuffle=True)
test_data  = DataLoader(dataset=test_set,batch_size=batch_size ,shuffle=False)

net = MSResNet(input_channel=input_channel, layers=[1, 1, 1, 1], n_output=49).cuda()
#optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9,weight_decay = 0.0005)
#optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
optimizer = torch.optim.Adam([{'params':
                                  filter(lambda p: p.requires_grad,
                                         net.parameters()),
                                  'lr': base_lr}],
                                lr=base_lr,
                                weight_decay=0.0005,
                                )
#criterion =	torch.nn.MSELoss()
criterion =	mseLoss()


losses = []
eval_losses = []
Best = 0.
checkpoint = 0
epoch_iters = np.int(len(train_set) / batch_size)
num_iters = Epoch * epoch_iters
test_features = test_features.view(test_features.size(0),input_channel,1)
for epoch in range(Epoch):
    train_loss = 0
    net.train()
    cur_iters = epoch * epoch_iters
    for i_iter, batch in enumerate(train_data, 0):
        tdata,tlabel = batch
        tdata = tdata.view(tdata.size(0),input_channel,1)
        y_ = net(tdata)
        #print(y_.shape)
        loss = criterion(y_, tlabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
        lr = adjust_learning_rate(optimizer,base_lr,num_iters,i_iter+cur_iters)

    losses.append(train_loss / len(train_data))
    #losses.append(math.sqrt(train_loss / len(train_data)))
    eval_loss = 0
    net.eval()  
    for edata, elabel in test_data:
        edata = edata.view(edata.size(0),input_channel,1)
        y_ = net(edata)
        loss = criterion(y_, elabel)
        eval_loss = eval_loss + loss.item()
    eval_losses.append(eval_loss / len(test_data))
    #eval_losses.append(math.sqrt(eval_loss / len(test_data)))
    
    y_pre = net(test_features)
    MSE = mse(y_pre.squeeze().detach().cpu().numpy(),test_labels.squeeze().detach().cpu().numpy())
    if MSE > Best:
        Best = MSE
        checkpoint = epoch
    print('epoch: {}, lr: {:.6f}, trainloss: {:.6f}, evalloss: {:.6f}, evalMSE: {:.6f}, BestMSE: {:.6f}'.format(epoch,lr, train_loss / len(train_data), eval_loss / len(test_data),MSE,Best))
