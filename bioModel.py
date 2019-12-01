# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

import torch
import torch.nn as nn

class bioModel(nn.Module):
    def __init__(self, words, seqLen=57, embeddingSize=300, hiddenSize=200, classes=3, num_layers=1, batch_size=32):
        super(bioModel, self).__init__()
        self.words = words
        self.seqLen = seqLen
        self.embeddingSize = embeddingSize
        self.hiddenSize = hiddenSize
        self.classes = classes
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(len(self.words) + 1, self.embeddingSize)

        self.biLstm = nn.LSTM(input_size=self.embeddingSize, hidden_size=self.hiddenSize, batch_first=True,
                              bidirectional=True)

        self.fc = nn.Linear(self.hiddenSize * 2, self.classes)

    def hidden_init(self):
        if self.biLstm.bidirectional:
            return (torch.rand(self.num_layers * 2, self.batch_size, self.hiddenSize).cuda(),
                    torch.rand(self.num_layers * 2, self.batch_size, self.hiddenSize).cuda())
        else:
            return (torch.rand(self.num_layers, self.batch_size, self.hiddenSize).cuda(),
                    torch.rand(self.num_layers, self.batch_size, self.hiddenSize).cuda())

    def forward(self, input):
        encoder_hidden, encoder_c = self.hidden_init()
        embeds = self.embedding(input)
        out, (encoder_hidden, encoder_c) = self.biLstm(embeds, (encoder_hidden, encoder_c))
        out = torch.softmax(self.fc(out), dim=2)
        return out
