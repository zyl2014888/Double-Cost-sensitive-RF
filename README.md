# Double-Cost-sensitive-RF
双重代价敏感随机森林算法
参考文献

Cost-sensitive feature selection using random forest: Selecting low-cost subsets of informative features 2016 《Knowledge-Based Systems》

算法改进

相比于上一章节分享的代价敏感随机森林而言，这次引入了特征选择和序贯分析。
参考文献的特征选择算法只是单纯的计算出一个特征代价向量使随机过程更具有倾向性，但并未考虑特征间的相对关系，并且在特征区分度不大时退化成普通的RF算法。
鉴于此，提出了三点改进：
1）在生成特征向量阶段引入序贯分析
2）在Gini系数上做了调整
3）在决策树集成阶段引入了代价敏感，选择代价少的前90%的决策树（经实验计算，选择50%~90%的决策树数量准确度没什么区别）
