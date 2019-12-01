# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

import torch
import torch.nn as nn


class relationModel(torch.nn.Module):
    def __init__(self, words, maxSeqL=56, embeddingSize=300, hiddenSize=200, classes=4036, num_layers=1, batch_size=32):
        super(relationModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.maxSeqL = 56
        self.embeddingSize = embeddingSize
        self.words = words
        self.classes = classes
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(len(self.words) + 1, self.embeddingSize)
        self.sentenceEncoder = nn.LSTM(input_size=self.embeddingSize, hidden_size=self.hiddenSize, batch_first=True,
                                       bidirectional=True, dropout=0)

        # self.fc = nn.Linear(2*self.maxSeqL*self.hiddenSize, self.classes)
        self.fc = nn.Linear(2*self.hiddenSize, self.classes)

    def initHidden(self):
        return (torch.rand(self.num_layers * 2, self.batch_size, self.hiddenSize).cuda(),
                torch.rand(self.num_layers * 2, self.batch_size, self.hiddenSize).cuda())

    def forward(self, problem):
        problem = self.embedding(problem)
        p_h, p_c = self.initHidden()
        # out, (_, _) = self.sentenceEncoder(problem, (p_h, p_c))
        # out = out.permute(0, 2, 1)
        # out = out.reshape(self.batch_size, 2*self.maxSeqL*self.hiddenSize).squeeze(dim=1)
        # out = torch.softmax(out, dim=1)
        _, (p_h, _) = self.sentenceEncoder(problem, (p_h, p_c))
        out = self.changeData(p_h)
        out = torch.softmax(self.fc(out), dim=1)
        return out

    def changeData(self, vector):
        vector = vector.permute(1, 0, 2)
        vector = vector.reshape(1, self.batch_size, 2 * self.hiddenSize).squeeze(dim=0)
        return vector