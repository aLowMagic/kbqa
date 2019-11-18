# # Time    : 2019/11/6 22:20
# # Author  : yangfan
# # Email   : thePrestige_yf@outlook.com
# # Software: PyCharm
# import pymysql
# import torch
# from torch import nn
# import torch.utils.data as DataLoad
# from torch.autograd import Variable
# import numpy as np
#
# class moduleTest(nn.Module):
#     def __init__(self, seqLength, words, inputSize=300, hiddenSize=200, num_layers=1, embSize=300, batchSize=32):
#         super(moduleTest, self).__init__()
#         self.inputSize = inputSize#输入的特征维数
#         self.hiddenSize = hiddenSize#隐层状态维数
#         self.num_layers = num_layers#RNN层数
#         self.maxInputLength = len(words)+2
#         self.embSize = embSize#embedding维数
#         self.batchSize = batchSize
#         self.words = words
#         self.seqLength = seqLength#最长句长
#
#         self.embedding = nn.Embedding(self.maxInputLength, self.embSize)
#         self.encoder0 = nn.LSTMCell(input_size=self.inputSize, hidden_size=self.hiddenSize//2)
#         self.encoder1 = nn.LSTMCell(input_size=self.inputSize, hidden_size=self.hiddenSize//2)
#         self.ner = nn.LSTMCell(input_size=self.hiddenSize, hidden_size=self.hiddenSize)
#         self.dnn0 = nn.Linear(self.hiddenSize, 800)
#         self.dnn1 = nn.Linear(800, 3)
#     def init_hidden(self, hiddenSize):
#         return torch.rand(self.batchSize, hiddenSize)
#
#     def forward(self, input):
#         h0 = self.init_hidden(self.hiddenSize//2)
#         h1 = self.init_hidden(self.hiddenSize//2)
#         h2 = self.init_hidden(self.hiddenSize)
#         c0 = self.init_hidden(self.hiddenSize//2)
#         c1 = self.init_hidden(self.hiddenSize//2)
#         c2 = self.init_hidden(self.hiddenSize)
#         #转化为embedding
#         data = []
#         input = self.embedding(input.clone().detach().long())
#         #双向LSTM获取隐向量
#         memOrder = []
#         for i in range(self.seqLength):
#             x = input[:, i, :]
#             h0, c0 = self.encoder0(x, (h0, c0))
#             memOrder.append(h0.unsqueeze(1))
#         memReverse = []
#         for i in range(self.seqLength-1, -1, -1):
#             x = input[:, i, :]
#             h1, c1 = self.encoder1(x, (h1, c1))
#             memReverse.append(h1.unsqueeze(1))
#         memReverse.reverse()
#         memOrder = torch.cat(memOrder, dim=1)
#         memReverse = torch.cat(memReverse, dim=1)
#         encoderRes = torch.cat((memOrder, memReverse), dim=2)
#
#         decoder = encoderRes.new_zeros(encoderRes.size())
#         for i in range(self.seqLength):
#             enc = encoderRes[:, i, :]
#             h2, c2 = self.ner(enc, (h2, c2))
#             decoder[:,i,:]=h2
#
#         o = torch.relu(self.dnn0(decoder))
#         o = torch.relu(self.dnn1(o))
#         return o
#
#
# def process(filePath):
#     maxLength = 0
#     words = set()
#     with open(filePath, 'r', encoding="utf8") as f:
#         lines = f.readlines()
#         for line in range(int(len(lines)/2)):
#             problem = lines[line]
#             maxLength = max(maxLength, len(problem))
#             for s in problem:
#                 words.add(s)
#     wordSize = len(words)
#     words = {word: i+1 for i, word in enumerate(words)}
#     return maxLength, wordSize, words
#
# def padding(maxLength, data):
#     outData = np.zeros(maxLength)
#     for d in range(int(len(data))):
#         outData[d] = data[d]
#     return outData
#
# def dataLoad(fileName, batchSize=32, num_workers=1):
#     seqLength, wordSize, words = process(fileName)
#     bioToNum = {"O": 0, "B": 1, "I": 2}
#     data = []
#     label = []
#     with open(fileName, 'r', encoding="utf8") as f:
#         lines = f.readlines()
#         for line in range(0, len(lines)//2, 2):
#             sent = lines[line].rstrip("\n")
#             bio = lines[line+1].rstrip("\n")
#             sent = padding(seqLength, [words[s] for s in sent])
#             bio = padding(seqLength, [bioToNum[b] for b in bio])
#             data.append(sent)
#             label.append(bio)
#     trainSetLength = int(len(data)*0.8)
#
#     trainSet = list(zip(data[:trainSetLength], label[:trainSetLength]))
#     testSet = list(zip(data[trainSetLength:], label[trainSetLength:]))
#     train = DataLoad.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers=num_workers)
#     test = DataLoad.DataLoader(dataset=testSet, batch_size=batchSize, shuffle=True, num_workers=num_workers)
#     return train, test, words, seqLength
#
# def judge(outt, label):
#     for s in range(label.size(0)):
#         if outt[s] != label[s]:
#             return False
#     return True
#
# def sentenceLength(input):
#     counter = input.size(0)
#     for j in range(counter):
#         if input[j] == 0:
#             return j
#     return 57
#
# if __name__=="__main__":
#     nerTxt = "./nlpcc2018.trainset/ner.txt"
#     trainSet, testSet, words, seqLength = dataLoad(nerTxt)
#     maxLength, wordSize, words = process(nerTxt)
#     module = moduleTest(seqLength=seqLength, words=words, inputSize=300, hiddenSize=200, num_layers=1, embSize=300, batchSize=32)
#     cirtertion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9)
#     for epoch in range(40):
#         for i, (inputs, labels) in enumerate(trainSet, 0):
#             inputs = torch.tensor(inputs, dtype=torch.float)
#             labels = torch.tensor(labels, dtype=torch.float)
#             inputs = Variable(inputs)
#             labels = labels.clone().detach().long()
#
#             optimizer.zero_grad()
#             outputs = module(inputs)
#             tot_loss = 0
#             for i in range(labels.size(0)):
#                 input = inputs[i]
#                 counter=input.size(0)
#                 for j in range(counter):
#                     if input[j] == 0:
#                         counter = j
#                         break
#                 out = outputs[i, 0:counter]
#
#                 target = labels[i, 0:counter]
#                 tot_loss += cirtertion(out, target)
#                 print(out)
#                 print(target)
#             tot_loss.backward()
#             optimizer.step()
#             #print('[%d] loss:%.3f' % (epoch + 1, tot_loss.data))
#         try:
#             torch.save(module, "checkpoint_epoch"+str(epoch+1)+".pkl")
#         except:
#             print("你不会保存模型")
#     module = torch.load('checkpoint_epoch20.pkl')
#     counter = 0
#     counterFalse = 0
#     for i, (inputs, labels) in enumerate(testSet, 0):
#         inputs = torch.tensor(inputs, dtype=torch.float)
#         labels = torch.tensor(labels, dtype=torch.float)
#         inputs = Variable(inputs)
#         labels = labels.clone().detach().long()
#         outputs = module(inputs)
#         for i in range(labels.size(0)):
#             input = inputs[i]
#             sentence_length = sentenceLength(input)
#             out = outputs[i, 0:sentence_length]
#             _, outt = out.max(dim=1)
#             label = labels[i, 0:sentence_length]
#             counter+=1
#             if not judge(outt, label):
#                 counterFalse+=1
#             else:
#                 print(label)
#                 print(outt)
#     print(counterFalse)
#     print(counter)
#     print("epoch%d error Rate: %d%%"%(30, 100*counterFalse/counter))
#     print("finished")