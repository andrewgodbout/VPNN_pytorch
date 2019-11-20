import torch
import time


# helper to compute accuracy of valid cycle
# y is the model output of size NxClasses
# targets is the labels of the inputs, of size N
# does top 1 error
def acc(y, targets):
    preds = torch.argmax(y, dim=1)
    return (preds == targets).float().mean()
"""
inputs:
model - some function that can take in a tensor, cuda tensor if available.
loss_funct - takes in predictions and either one hot vectors or labels to compute loss
train_dl - training dataloader
valid_dl - validation dataloader
opt - some torch.nn.optim that has model parameters
one_hot_size - length of one hot vector for labels
expand_targets - passes labels as one hot vectors to loss_funct if true, raw labels otherwise. default:True
pred_funct - some function to interpret model outputs. If none no change. default:None
mode - specifies behaviour. default '' to do full pass then test accuracy
    'step' to do one batch and step, no accuracy
    'test' to just test accuracy
"""
# training cycle 
def fit(model, loss_funct, train_dl, valid_dl, opt, epochs, one_hot_size, mode=''):
    losses=[]
    accs=[]
    loss=torch.tensor(float('nan'))
    for epoch in range(epochs):
        t = time.time()
        # training
        for x, y in train_dl:
            if mode=='test':
                break
            # forward pass 
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            pred = model(x) 

            # loss and backward pass
            loss = loss_funct(pred, y)  
            
            # if loss became nan, end training
            if torch.isnan(loss):
                print('loss returned nan at training batch:', len(losses), int(time.time()-t), 's')
                data = dict(zip(['losses','accs'],[losses, accs]))
                return data
            
            loss.backward()
            # data collection
            with torch.no_grad():
                losses.append(loss.item())
            # param updating         
            opt.step()  
            opt.zero_grad() 
            
            if mode=='step':
                return loss
            
            
        # verification
        with torch.no_grad():
            epoch_acc = []
            for xv, yv in valid_dl:
                if torch.cuda.is_available():
                     xv = xv.cuda()
                     yv = yv.cuda()
                pred = model(xv)
                epoch_acc.append(acc(pred, yv).item())
            # mean results from each cycle
            a=sum(epoch_acc)/len(epoch_acc)*100
            accs.append(a)
            print(epoch, "{:.2f}".format(a), '%', int(time.time()-t), 's', "training loss: ", loss.item())
        if mode=='test':
            return
    data = dict(zip(['losses','accs'],[losses, accs]))
    return data


