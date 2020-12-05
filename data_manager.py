import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchtext import data

import math
import time

SEED=12
BATCH_SIZE=128
LEARNING_RATE=1e-3
EMBEDDING_DIM=100

#为CPU设置随机种子
torch.manual_seed(SEED)

#使用torchtext处理训练数据
TEXT = data.Field(tokenize=lambda x: x.split(), lower=True)#Field对象相当于占位，以及对占位内容做一些处理
LABEL = data.LabelField(dtype=torch.float)

def get_dataset(corpus_path,text_field,label_field):
    fields=[('text',text_field),('label',label_field)]#这里类似规定一个格式
    examples=[]

    with open(corpus_path) as f:
        li=[]
        while True:
            content=f.readline().strip()
            if not content: #若为空行
                if not li: #而且为空列表，表示数据读完
                    break
                label=li[0][10]
                text = li[1][6:-7]
                examples.append(data.Example.fromlist([text,label],fields))#相当于从将list的内容实例化field对象
                li=[]
            else:
                li.append(content)
    return examples,fields
train_examples,train_fields=get_dataset('trains.txt',TEXT,LABEL)
dev_examples,dev_fields=get_dataset('dev.txt',TEXT,LABEL)
test_examples,test_fields=get_dataset('tests.txt',TEXT,LABEL)

train_data=data.Dataset(train_examples,train_fields)
dev_data = data.Dataset(dev_examples, dev_fields)
test_data = data.Dataset(test_examples, test_fields)

#创建vocabulary
TEXT.build_vocab(train_data,max_size=5000,vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

#创建iterators,每个iteration都会返回一个batch的examples
train_iterator,dev_iterator,test_iterator=data.BucketIterator.splits(
    (train_data,dev_data,test_data),
    batch_size=BATCH_SIZE,
    sort=False)



