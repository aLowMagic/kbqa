# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

import torch

class thisDataSet(torch.utils.data.Dataset):
    def __init__(self, problem, relation, bioLabel, relationLabel):
        self.problem = problem
        self.relation = relation
        self.bioLabel = bioLabel
        self.relationLabel = relationLabel
    def __getitem__(self, index):
        problem, relation, bioLabel, relationLabel = self.problem[index], self.relation[index], self.bioLabel[index], self.relationLabel[index]
        return problem, relation, bioLabel, relationLabel
    def __len__(self):
        return len(self.problem)