# -*- coding: utf-8 -*-
"""
@Env:Python2.7
@Time: 2019/10/24 13:31
@Author: zhaoxingfeng
@Function：Random Forest（RF），随机森林二分类
@Version: V1.1
参考文献：
[1] UCI. wine[DB/OL].https://archive.ics.uci.edu/ml/machine-learning-databases/wine.
"""
import pandas as pd
import numpy as np
import random
import math
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)
import collections
import time
import warnings
warnings.filterwarnings("ignore")
def Cost_feature(Cost):
    #赋值
    P=[]
    r=[]
    for i in range(len(Cost)):
        r.append(float(1/Cost[i]))
    a=sum(r)
    for i in range(len(Cost)):
        P.append(float(format(float(r[i]/a),'.6f')))
    return P




# 定义一棵决策树
class Tree(object):
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None



    # 通过递归决策树找到样本所属叶子节点
    def calc_predict_value(self, dataset):
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    # 以json形式打印决策树，方便查看树结构
    def describe_tree(self):
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree="sqrt", subsample=1.0, random_state=None,Cost=[],iteration=False):
        self.n_estimators = n_estimators #决策树个数
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree  # 列采样
        self.subsample = subsample  # 行采样
        self.random_state = random_state
        self.trees = dict()
        self.feature_importances_ = dict()
        self.Cost=Cost
        self.AC_Treee_List=[]
        self.P=[]
        #记录准确率高于指定阈值的特征
        self.features=[]
        self.iteration=iteration

    # 修改代价向量
    def Clu_CostMatrix(self, tree_position):
        for i in range(0, self.colsample_bytree):
            self.Cost[int(self.features[tree_position][i])]+=1

    #初始化得到代价向量
    def Get_Cost(self,dataset,label,train_index):
        res = []
        count=len(label)
        for stage,tree in self.trees.items():
            pred_list = []
            for index, row in dataset.iterrows():
                pred_list.append(tree.calc_predict_value(row))
            #里边装的是每棵树对所有数据的分类
            res.append(pred_list)
        #计算每棵树的准确率
        for j in range(len(res)):
            acc = 0
            for i in range(count):
                if res[j][i]==int(label[i+train_index]):
                    acc+=1
            acc=float(acc/len(label))
            if acc>0.99:
                self.Clu_CostMatrix(j)

        # 初始化得到代价向量

    def Get_Cost_Test(self,tree,dataset, label, train_index,features):
        res = []
        pred_list = []
        count = len(label)
        for index, row in dataset.iterrows():
            pred_list.append(tree.calc_predict_value(row))
        # 计算每棵树的准确率
        acc = 0
        for j in range(len(pred_list)):

            if pred_list[j] == int(label[j + train_index]):
                acc += 1
        acc = float(acc / len(label))
        if acc > 0.95:
            for i in features:
                self.Cost[int(i)]+=5

    def fit(self, dataset, targets,traindata,trainlabel,train_index):
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')
        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)
        # 两种列采样方式
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)
        for stage in range(self.n_estimators):
            #计算代价向量的倒数向量
            self.P=Cost_feature(self.Cost)
            # bagging方式随机选择样本和特征
            random.seed(random_state_stages[stage])
            subset_index = random.sample(range(len(dataset)), int(self.subsample * len(dataset)))

            #随机选择的特征
            subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
            #随机特征对应的数据
            dataset_copy = dataset.loc[subset_index, subcol_index].reset_index(drop=True)

            #随机特征对应的标签
            targets_copy = targets.loc[subset_index, :].reset_index(drop=True)


            tree = self._fit(dataset_copy, targets_copy, depth=0)
            self.trees[stage] = tree

            #如果迭代改变Cost
            if self.iteration:
                self.Get_Cost_Test(tree,traindata,trainlabel,train_index,subcol_index)
            else:
                self.features.append(subcol_index)


    # 递归建立决策树
    def _fit(self, dataset, targets, depth):
        # 如果该节点的类别全都一样/样本小于分裂所需最小样本数量，则选取出现次数最多的类别。终止分裂
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._fit(left_dataset, left_targets, depth+1)
                tree.tree_right = self._fit(right_dataset, right_targets, depth+1)
                return tree
        # 如果树的深度超过预设值，则终止分裂
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    # 选择最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益
    def choose_best_feature(self, dataset, targets):
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])
                split_gain=self.P[int(feature)]*split_gain
                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    # 选择样本中出现次数最多的类别作为叶子节点取值
    @staticmethod
    def calc_leaf_value(targets):
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    # 分类树采用基尼指数来选择最优分裂点
    @staticmethod
    def calc_gini(left_targets, right_targets):
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # 统计每个类别有多少样本，然后计算gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                a=(1-prob)**1
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    # 根据特征和阈值将样本划分成左右两份，左边小于等于阈值，右边大于阈值
    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    # 输入样本，预测所属类别
    def predict(self, dataset):
        res = []
        for index, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，选取出现次数最多的结果作为最终类别
            for stage, tree in self.trees.items():
                pred_list.append(tree.calc_predict_value(row))
            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)

    #得到新的决策树集合
    def Clu_Cost(self,dataset,label,train_index,treeCount):
        res = []
        count=len(label)
        for stage,tree in self.trees.items():
            pred_list = []
            for index, row in dataset.iterrows():
                pred_list.append(tree.calc_predict_value(row))
            #里边装的是每棵树对所有数据的分类
            res.append(pred_list)
        #用来装新的树的列表
        CostTreeList=[]

        i=0
        #计算每个树的TP，FN值
        for treePre in  res:
            FP=0;FN=0;ac=0
            for j in range(len(label)):
                if treePre[j]!=int(label[train_index+j]):
                    if treePre[j]==1:#预测是1，实际是2
                        FP+=1
                    else:#预测是2，实际是1
                        FN+=1

            ac=float((FP*1+FN*16)/count)
            l=[]
            l.append(ac)
            l.append(self.trees[i])
            CostTreeList.append(l)
            i+=1

        #得到树AC升序列表
        CostTreeList=self.SortedTree(CostTreeList)




        #得到新的决策树集合
        self.trees={}
        for j in range(0,treeCount):
            self.trees[j]=(CostTreeList[j][1])
        self.n_estimators=treeCount

    #决策树排序
    def SortedTree(self,CostDict):
        for i in range(0,self.n_estimators-1):
            for j in range(0,self.n_estimators-1-i):
                if CostDict[j][0]>CostDict[j+1][0]:
                    t=[]
                    t.append(CostDict[j+1][0])
                    t.append(CostDict[j+1][1])
                    CostDict[j+1][0]=CostDict[j][0]
                    CostDict[j+1][1]=CostDict[j][1]
                    CostDict[j][0]=t[0]
                    CostDict[j][1]=t[1]
        return CostDict


    def RecalPredic(self,dataset,label,train_index):
        res = []
        for index, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，选取出现次数最多的结果作为最终类别
            for stage, tree in self.trees.items():
                pred_list.append(tree.calc_predict_value(row))
            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])

        TP=0;FN=0

        for i in range(len(res)):
            if int(label[train_index+i])==2:
                if int(res[i])==int(label[train_index+i]):
                    TP+=1
                else:
                    FN+=1
            else:
                continue
        print(TP,FN,float(TP/(TP+FN)))



#约分代价向量
def Approximate_score(Cost):
    if 1  in Cost:
        return Cost
    Cost=[i-1 for i in Cost]

    while 1:
        count=0
        for i in Cost:
            if i%5==0:
                count+=1
        if count==len(Cost):
            Cost=[int(a/5) for a in Cost]
        else:
            break
    return Cost
#创建初始Cost
def InitCost(m):
    Cost=[]
    for i in range(m):
        Cost.append(1)
    return Cost
if __name__ == '__main__':
    df = pd.read_csv("source/wine.txt")
    #frac=0.8 表示抽取0.8的数据
    df = df[df['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)
    Cost=InitCost(df.shape[1]-1)
    '''
        n_emtimators:树的数量
    '''
    train_count = int(0.7 * len(df))
    #Cost=Cost_feature(Cost)
    clf1 = RandomForestClassifier(n_estimators=90,
                                 max_depth=5,
                                 min_samples_split=6,#分裂所需的最小增益，小于该值分裂停止
                                 min_samples_leaf=2,#控制几分类
                                 colsample_bytree="sqrt",
                                 subsample=0.8,#行采样率
                                 random_state=66,#复现上一次的结果。因为每生成一棵树都要采样，设置固定的随机种子就可以让上一次和这一次的样本一致
                                 Cost=Cost,
                                 iteration=True)

    clf1.fit(df.ix[:train_count, :-1], df.ix[:train_count, 'label'],df.ix[train_count:, :-1],df.ix[train_count:, -1],train_count)
    #clf1.Get_Cost(df.ix[train_count:, :-1],df.ix[train_count:, -1],train_count)


    clf1.Cost=Approximate_score(clf1.Cost)
    print(clf1.Cost)
    #得到代价矩阵后重新建树

    s1 = time.time()
    clf = RandomForestClassifier(n_estimators=115,
                                 max_depth=5,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=66,
                                 Cost=clf1.Cost)
    clf.fit(df.ix[:train_count, :-1], df.ix[:train_count, 'label'],df.ix[train_count:, :-1],df.ix[train_count:, -1],train_count)

    from sklearn import metrics



    print('准确率：{:.2f}%'.format(
        100*metrics.accuracy_score(df.ix[train_count:, 'label'], clf.predict(df.ix[train_count:, :-1]))))
    print('召回率：{:.2f}%'.format(
        100 * metrics.recall_score(df.ix[train_count:, 'label'], clf.predict(df.ix[train_count:, :-1]))))

    clf.Clu_Cost(df.ix[train_count:, :-1], df.ix[train_count:, -1], train_count,100)
    s2 = time.time()
    print('准确率：{:.2f}%'.format(
        100 * metrics.accuracy_score(df.ix[train_count:, 'label'], clf.predict(df.ix[train_count:, :-1]))))
    print('召回率：{:.2f}%'.format(
        100*metrics.recall_score(df.ix[train_count:, 'label'], clf.predict(df.ix[train_count:, :-1]))))
    print('运行时间:{:.2f}'.format(s2-s1))
    clf.RecalPredic(df.ix[train_count:, :-1], df.ix[train_count:, -1], train_count)

'''
    #第一个索引是train_count'''




