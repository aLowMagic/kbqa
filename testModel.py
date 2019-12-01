# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

from relationModel import relationModel
from bioModel import bioModel
from DataWrap import DataWrap
import torch
import torch.nn as nn

def getSubject(problem, bio):
    bio = bio[0]
    i = 0
    while i<len(bio) and bio[i]==0:
        i+=1
    if i>len(problem):
        return "无法识别"
    subject = ""
    while bio[i]!=0 and i<len(problem):
        subject+=problem[i]
        i+=1
    return subject


if __name__=="__main__":
    txtPath = "./relation_match_14.txt"
    dataWrap = DataWrap()
    bio = bioModel(words=dataWrap.getWords(), batch_size=1)
    bio.load_state_dict(torch.load("./bioCheckPoint.pkl"))
    bio.cuda().eval()
    rel = relationModel(words=dataWrap.getWords(), classes=len(dataWrap.getRelations()), batch_size=1)
    rel.load_state_dict(torch.load("./relMCheckpoint.pkl"))
    rel.cuda().eval()
    class_to_relation = {i: words for words, i in dataWrap.getRelations().items()}
    while True:
        problem = input("请输入问题：\n")
        input_problem = dataWrap.testModel(problem)
        subject = "无法识别"
        for i, (bio_problem, _, _) in enumerate(input_problem):
            bio_problem = torch.tensor(bio_problem, dtype=torch.long).cuda()
            _, output = bio(bio_problem).max(dim=2)
            subject = getSubject(problem, output.cpu().numpy().tolist())
        print("subject = "+subject)
        if subject!="无法识别":
            for i, (relProblem, _, _) in enumerate(input_problem):
                relProblem = torch.tensor(relProblem, dtype=torch.long).cuda()
                _, output = rel(relProblem).max(dim=1)
                classify = output.cpu().numpy().tolist()[0]
                print("relation："+class_to_relation[classify])
                print("")
        else:
            print("")


