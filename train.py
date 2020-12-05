from model import *

#计算准确率
def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))#torch.round进行四舍五入
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

#训练函数
def train(rnn,iterator,optimizer,criteon):
    avg_loss=[]
    avg_acc=[]
    rnn.train() #表示进入训练模式

    for i,batch in enumerate(iterator):
#        batch.text=batch.text.to(device)#使用GPU
#        batch.label=batch.label.to(device)

        pred=rnn(batch.text).squeeze() #[batch,1]->[batch]
        loss=criteon(pred,batch.label)
        acc=binary_acc(pred,batch.label).item() #计算每个batch的准确率

        avg_loss.append(loss.item())
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc=np.array(avg_acc).mean()
    avg_loss=np.array(avg_loss).mean()
    return avg_loss,avg_acc

#评估函数
def evaluate(rnn,iterator,criteon):
    avg_loss=[]
    avg_acc=[]
    rnn.eval()

    with torch.no_grad():
        for batch in iterator:
#            batch.text = batch.text.to(device)  # 使用GPU
#            batch.label = batch.label.to(device)

            pred=rnn(batch.text).squeeze()
            loss=criteon(pred,batch.label)
            acc=binary_acc(pred,batch.label).item()
            avg_loss.append(loss.item())
            avg_acc.append(acc)

    avg_loss=np.array(avg_loss).mean()
    avg_acc=np.array(avg_acc).mean()
    return avg_loss,avg_acc

#训练模型，并打印模型表现
best_valid_acc=float('-inf')

for epoch in range(10):
    start_time=time.time()

    train_loss, train_acc = train(rnn, train_iterator, optimizer, criteon)
    dev_loss, dev_acc = evaluate(rnn, dev_iterator, criteon)

    end_time = time.time()

    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    if dev_acc > best_valid_acc:  # 只要模型效果变好，就保存
        best_valid_acc = dev_acc
        torch.save(rnn.state_dict(), 'wordavg-model.pt')
    print('Epoch: {} | Epoch Time: {}m {}s'.format(epoch + 1,epoch_mins,epoch_secs))
    print('\tTrain Loss: {} | Train Acc: {}%'.format(train_loss,train_acc * 100))
    print('\t Val. Loss: {} |  Val. Acc: {}%'.format(dev_loss,dev_acc * 100))

#用保存的模型参数预测数据
rnn.load_state_dict(torch.load("wordavg-model.pt"))
test_loss, test_acc = evaluate(rnn, test_iterator, criteon)
print('Test. Loss: {} |  Test. Acc: {}%'.format(test_loss,test_acc*100))