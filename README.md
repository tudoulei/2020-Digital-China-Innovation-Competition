# 2020数字中国创新大赛-数字政府赛道

智能算法赛：智慧海洋建设

### 队伍基本信息
队伍名：用欧气驱散疫情

### 最终成绩

![](https://github.com/tudoulei/2020-Digital-China-Innovation-Competition/img/4.png)

### 思路

以建立手工特征，分箱子特征和多种word2vec特征为基础，使用lightgbm单模型进行训练五折得到最终结果

### 环境
- 操作系统 win10
- 运行pip install -r requirements.txt ,即可检查库是否存在

### 代码加载路径
- data文件夹，数据放到这个文件夹即可
- 如果出现问题，打开datapre.py文件修改12，13，14行代码

### 代码运行
python main.py

### 预测结果输出位置
- 预测结果会自动创建submit文件夹，
- 【result_0.92512_2020-02-21-15-57-24.csv】格式的为上交文件，
- 【PROB_result_0.92512_2020-02-21-16-48-55.csv】的为预测概率文件

###  特殊注意
- 若需要重新训练，word2vec重新生成可能出现不能复现原特征的结果，需要保证环境变量中新建一个PYTHONHASHSEED变量，并设定为0，但此方法也不一定保证完整复现一模一样的w2v特征
- 可参考 https://blog.csdn.net/weiyongle1996/article/details/80248352
- 使用保存好得w2v模型并不影响

### 队友的代码
https://github.com/CQLLL/2020DCIC

### 渔船管理系统界面

![](https://github.com/tudoulei/2020-Digital-China-Innovation-Competition/img/1.png)

![](https://github.com/tudoulei/2020-Digital-China-Innovation-Competition/img/2.png)

![](https://github.com/tudoulei/2020-Digital-China-Innovation-Competition/img/3.png)

