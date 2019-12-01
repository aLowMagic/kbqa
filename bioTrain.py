# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

from DataWrap import DataWrap
import torch
import torch.nn as nn
from bioModel import bioModel

if __name__=="__main__":
    cuda_gpu = torch.cuda.is_available()
    torch.cuda.set_device(2)
    txtPath = "./relation_match_14.txt"
    dataWrap = DataWrap(batchSize=32, txtPath=txtPath)
    batchSize = 32
    model = bioModel(words=dataWrap.getWords(), batch_size=batchSize)
    model = model.cuda()
    cirtertion = nn.CrossEntropyLoss()
    cirtertion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if cuda_gpu:
        for epoch in range(10):
            for i, (bioData, _, bioLabel, _) in enumerate(dataWrap.dataProcess()):
                bioData = torch.tensor(bioData, dtype=torch.long).cuda()
                bioLabel = torch.tensor(bioLabel, dtype=torch.long).cuda()
                optimizer.zero_grad()
                output = model(bioData)
                output = output.permute(0, 2, 1)
                loss = cirtertion(output, bioLabel)
                loss.backward()
                optimizer.step()
            print("epoch %d is finished"%epoch)
        torch.save(model.state_dict(), "./bioCheckPoint.pkl")