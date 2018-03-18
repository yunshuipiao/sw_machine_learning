from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, \
    classification_report, \
    confusion_matrix, \
    mean_absolute_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


if __name__ == '__main__':
    # base example
    # iris = datasets.load_iris()
    # X, y = iris.data[:, :2], iris.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    # print(accuracy_score(y_test, y_pred))

    # 详细流程
    #step1: 加载数据
    X = np.random.random((10, 5))
    y = np.array(['M', 'M', 'F', 'F', 'M', 'F', 'M', 'M','F','F'])



    # step2: 切分训练数据和测试数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    #step3： 预处理数据

    #标准化 standardization
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # standardized_X = scaler.transform(X_train)
    # standardized_X_test = scaler.transform(X_test)


    #规范化 Normalization：规范化是将不同变化范围的值映射到相同的固定范围，常见的是[0,1]
    # scaler = preprocessing.Normalizer().fit(X_train)
    # normalized_X = scaler.fit(X_train)
    # normalized_X_test = scaler.fit(X_test)

    # 二值化
    # binarizer = preprocessing.Binarizer(threshold=0.5).fit(X) #阈值
    # binaty_X = binarizer.transform(X)
    # print(binaty_X)

    # 标签编码
    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(y)

    #缺失值处理
    #imputing missing values
    # imp = preprocessing.Imputer(missing_values=0, strategy='mean', axis=0)
    # imp.fit_transform(X_train)

    #生成多项式特征
    # poly = preprocessing.PolynomialFeatures(5)
    # poly.fit_transform(X)

    #step4: 创建模型

    #监督学习估计器 supervised learning estimator
    ##线性回归
    # lr = LinearRegression(normalize=True)

    ##支持向量机 SVM
    # svc = SVC(kernel='linear')

    ## 朴素贝叶斯 Naive Bayes
    # gnb = GaussianNB()

    ##knn
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)

    #无监督学习估计器 unsupervised learning estimator
    ##主成分分析  principal component Analysis(PCA)
    # pca = PCA(n_components=0.95)

    ##K Means
    # k_means = KMeans(n_clusters=3, random_state=0)


    #step4: 模型训练

    ##监督学习
    # lr.fit(X, y)
    knn.fit(X_train, y_train)
    # svc.fit(X_train, y_train)

    ##无监督学习
    # k_means.fit(X_train)
    # pca_modal = pca.fit_transform(X_train)

    #step5: 预测
    ##无监督学习
    # y_pred = lr.predict(X_test)
    # y_pred = knn.predict_proba()

    ##监督学习
    # y_pred = k_means.predict(X_test)

    #step6: 评估模型
    ## 分类
    ### 准确率分数
    # knn.sroce(X_test, y_test)
    # accuracy_score(y_test, y_pred)

    ###  Classification Report
    # classification_report(y_test, y_pred)

    ### 混淆矩阵
    # confusion_matrix(y_test, y_pred)

    ## 回归
    ### 平均绝对误差 mean absolute error
    # y_true = [3, -0.5, 2]
    # mean_absolute_error(y_true, y_pred)
    ### 均方差
    ### R平方 score

    ##聚类
    ### 调整随机指标
    ### 同质
    ### V-measure

    ##交叉验证
    # scores = cross_val_score(knn, X_train, y_train, cv=4)


    #step7: 调整模型
    ## 格点搜索
    # params = {'n_neighbors': np.arange(1, 3),
    #             'metric': ["euclidean", 'cityblock']
    #           }
    # grid = GridSearchCV(estimator=knn, param_grid=params)
    # grid.fit(X_train, y_train)
    # print(grid.best_score_)
    # print(grid.best_estimator_.n_neighbors)














