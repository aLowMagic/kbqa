# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

from DataWrap import DataWrap
import torch
import torch.nn as nn
from relationModel import relationModel

if __name__=="__main__":
    cuda_gpu = torch.cuda.is_available()
    torch.cuda.set_device(2)
    if cuda_gpu:
        txtPath = "./relation_match_14.txt"
        dataWrap = DataWrap(batchSize=32, txtPath=txtPath)
        batchSize = 32
        classes = len(dataWrap.getRelations())
        model = relationModel(words=dataWrap.getWords(), classes=classes, batch_size=batchSize)
        model = model.cuda()
        cirtertion = nn.CrossEntropyLoss()
        cirtertion.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(5):
            for i, ((problem, _, _, relationLabel)) in enumerate(dataWrap.dataProcess()):
                problem = torch.tensor(problem, dtype=torch.long).cuda()
                relationLabel = torch.tensor(relationLabel, dtype=torch.long).cuda()
                output = model(problem)
                optimizer.zero_grad()
                loss = cirtertion(output, relationLabel)
                loss.backward()
                optimizer.step()

            print("epoch %d finished" % epoch)
        torch.save(model.state_dict(), "relMCheckpoint_5.pkl")