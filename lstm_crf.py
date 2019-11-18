# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import torch.nn as nn
import numpy as np
from torchcrf import CRF
import re

class model(nn.Module):
    def __init__(self, words, seqLen=57, embeddingSize=300, hiddenSize=200, classes=3, num_layers=1, batch_size=32):
        super(model, self).__init__()
        self.words = words
        self.seqLen = seqLen
        self.embeddingSize = embeddingSize
        self.hiddenSize = hiddenSize
        self.classes = classes
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(len(self.words)+1, self.embeddingSize)

        self.biLstm = nn.LSTM(input_size=self.embeddingSize, hidden_size=self.hiddenSize, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(self.hiddenSize, self.classes)

        # self.crfEmissions = torch.randn(self.batch_size, self.seqLen, self.classes)
        # self.crf = CRF(num_tags=self.classes, batch_first=True)

    def hidden_init(self):
        if self.biLstm.bidirectional:
            return (torch.rand(self.num_layers*2, self.batch_size, self.hiddenSize),
                    torch.rand(self.num_layers*2, self.batch_size, self.hiddenSize))
        else:
            return (torch.rand(self.num_layers, self.batch_size, self.hiddenSize),
                    torch.rand(self.num_layers, self.batch_size, self.hiddenSize))

    def forward(self, input):
        encoder_hidden, encoder_c = self.hidden_init()
        #encoder_c = self.hidden_init()
        embeds = self.embedding(input)
        out, (encoder_hidden, encoder_c) = self.biLstm(embeds, (encoder_hidden, encoder_c))
        out = torch.softmax(self.fc(out), dim=2)
       # print(out)
       # print(out.size())
        # print(outIndex.size())
        # print(labels.size())
        # out = self.crf.forward(self.crfEmissions, outIndex)
        # ground = self.crf.forward(self.crfEmissions, label)
        # print(out)
        # print(ground)
        return out



def process(filePath):
    maxLength = 0
    words = set()
    with open(filePath, 'r', encoding="utf8") as f:
        lines = f.readlines()
        for line in range(int(len(lines)/2)):
            problem = lines[line]
            maxLength = max(maxLength, len(problem))
            for s in problem:
                words.add(s)
    wordSize = len(words)
    words = {word: i+1 for i, word in enumerate(words)}
    return maxLength, wordSize, words

def padding(maxLength, data):
    outData = np.zeros(maxLength)
    for d in range(int(len(data))):
        outData[d] = data[d]
    return outData

def dataLoad(fileName, batchSize=32, num_workers=1):
    seqLength, wordSize, words = process(fileName)
    bioToNum = {"O": 0, "B": 1, "I": 2}
    data = []
    label = []
    senLength = []
    with open(fileName, 'r', encoding="utf8") as f:
        lines = f.readlines()
        for line in range(0, len(lines)//2, 2):
            sent = lines[line].rstrip("\n")
            bio = lines[line+1].rstrip("\n")
            sentP = padding(seqLength, [words[s] for s in sent])
            bioP = padding(seqLength, [bioToNum[b] for b in bio])
            data.append(sentP)
            label.append(bioP)
    wholeSet = list(zip(data, label))
    random.shuffle(wholeSet)
    trainSetLength = int(len(wholeSet)*0.8)
    trainSet = wholeSet[0:trainSetLength]
    trainSet = DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers=num_workers, drop_last=True)
    testSet = wholeSet[trainSetLength:]
    testSet = DataLoader(dataset=testSet, batch_size=batchSize, shuffle=True, num_workers=num_workers, drop_last=True)
    return trainSet, testSet, words, seqLength
def judgeTrue(data, label):
    for i in range(len(data)):
        if data[i]!=label[i]:
            return False
        else:
            print(data, label)
    return True

def testNer(sentence, model, words, seqLen):
    clearSentence = re.sub("[^0-9A-Za-z\u4e00-\u9fa5]", "", sentence)
    oneHotSentence = [words[i] for i in clearSentence]
    oneHotSentence = [padding(seqLen, oneHotSentence), padding(seqLen, [0 for _ in range(seqLen)])]
    testSet = DataLoader(dataset=oneHotSentence, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    for inputs, _ in testSet:
        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = model(inputs)
        print(outputs)


if __name__=="__main__":
    batchSize = 32
    trainSet, testSet, words, seqLength = dataLoad("./nlpcc2018.trainset/ner.txt", batchSize=batchSize)
    model = model(words=words, seqLen=seqLength, batch_size=batchSize)
    cirtertion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(10):
        tot_loss=0
        for i, (inputs, labels) in enumerate(trainSet, 0):
            inputs = torch.tensor(inputs, dtype=torch.long)
            labels = torch.tensor(np.array(labels), dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            tot_loss = cirtertion(outputs.permute(0, 2, 1), labels)
            tot_loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "./checkpoint_epoch.pkl")
        print(tot_loss)
    print("training finshed")
    counter = 0
    counterFalse = 0
    labels = []
    inputs = []
    outputs = []
    try:
        model = model(words=words, seqLen=seqLength, batch_size=batchSize)
        model.load_state_dict("./checkpoint_epoch.pkl")
        testNer("", model, words, seqLength)
    except:
        print("!")
    for i, (inputs, labels) in enumerate(testSet, 0):
        inputs = torch.tensor(inputs, dtype=torch.long)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        outputs = model(inputs)
        for output in range(batchSize):
            counter+=1
            _, pri = outputs[output].max(dim=1)
            if judgeTrue(pri, labels[output]):
                counterFalse+=1
    print("error rate: %d%%"%(100*counterFalse/counter))
