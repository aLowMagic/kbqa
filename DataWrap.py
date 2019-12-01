# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

import numpy as np
from torch.utils.data import DataLoader
from dataset import thisDataSet
import re



class DataWrap:

    def __init__(self, batchSize=32, txtPath="./relation_match_14.txt"):
        self.txtPath = txtPath
        self.batchSize = batchSize
        self.maxSentenceLength = 0
        self.maxSubjectLength = 0
        self.maxRelationLength = 0
        self.words = {"_": 1}
        self.relations = {}
        self.baseInformation()

    def getWords(self):
        return self.words

    def getRelations(self):
        return self.relations

    def testModel(self, problem):
        problem = re.sub("[^0-9A-Za-z\u4e00-\u9fa5]", "", problem)
        problem = self.paddingProblem(problem)
        label = np.zeros(1)
        res = list(zip([problem], [label], [label]))
        return DataLoader(dataset=res, batch_size=1)

    def dataProcess(self):
        with open(self.txtPath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            bioData = []
            relationData = []
            bioLabel = []
            relationLabel = []
            for line in lines:
                problem, subject, relation = line.split("\t")
                relation = relation.strip("\n")
                bio = self.labelBio(problem, subject)
                problem = self.paddingProblem(problem)
                padded_relation = self.paddingRelation(relation)
                bioData.append(problem)
                bioLabel.append(bio)
                relationData.append(padded_relation)
                relationLabel.append(np.array(self.relations[relation]))
            res = list(zip(bioData, relationData, bioLabel, relationLabel))
            return DataLoader(dataset=res, batch_size=self.batchSize, shuffle=True, num_workers=5, drop_last=True)

    def paddingRelation(self, relation):
        res = np.zeros(self.maxRelationLength)
        for  i in range(len(relation)):
            res[i] = self.words[relation[i]]
        return res

    def labelBio(self, problem, subject):
        res = np.zeros(self.maxSentenceLength)
        index = problem.find(subject)
        res[index] = 1
        for i in range(1, len(subject), 1):
            res[index+i] = 2
        return res

    def paddingProblem(self, problem):
        res = np.zeros(self.maxSentenceLength)
        for p in range(len(problem)):
            res[p] = self.words[problem[p]]
        return res

    def baseInformation(self):
        with open(self.txtPath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                problem, subject, relation = line.split("\t")
                relation = relation.strip("\n")
                self.maxSentenceLength = max(self.maxSentenceLength, len(problem))
                self.maxSubjectLength = max(self.maxSubjectLength, len(subject))
                self.maxRelationLength = max(self.maxRelationLength, len(relation))
                self.insertToWords(problem)
                self.insertToWords(subject)
                self.insertToWords(relation)
                if not self.relations.__contains__(relation):
                    self.relations[relation] = len(self.relations)


    def insertToWords(self, seq):
        for s in seq:
            if not self.words.__contains__(s):
                self.words[s] = len(self.words) + 1