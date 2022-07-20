# CompSci 711 - Intro to ML
# Name: Jonathan Nguyen 
# Date: 04/10/2022 
# Title: Assignment 2
# Version: 1.0 

# Task: 
#  In this assignment you will use sklearn to compare multiple machine learning methods on a regression 
#  task using learning curves and computational times.
#  1. Please go over A2.pptx slides.
#  2. Choose a dataset for regression task. You can choose it from OpenML (see the slides about Regression 
#  Datasets). The dataset must have
#   - At least one nominal feature
#   - At least 1000 examples
#  Choose a dataset in which the number of examples are far more than the number of features.
#  If the dataset has missing feature values then use the code from the Data Exploration slides to impute them.
#  Handle nominal features as described in the A2.pptx slides.
#  3. Build the following models and compare their learning curves and computational times.
#   - DecisionTreeRegressor in sklearn.tree
#       * Tune min_samples_leaf
#   - KNearestNeighborsRegressor in sklearn.neighbors
#       * Tune n_neighbors
#   - LinearRegression in sklearn.linear_model
#   - Support Vector Machine regressor: SVR in sklearn.svm
#   - Bagged decision tree regressor: BaggedRegressor  in sklearn.ensemble
#   - DummyRegressor in sklearn.dummy (as a baseline, it predicts the average value from the training data)
#  4. For each model, obtain rmse with increasing amount of training data using 10-fold  cross-validation. 
#  Plot a graph showing the learning curves of all the above models.
#  5. In a table, compare rmse of all the models obtained using 10-fold cross-validation with full training data 
#  (i.e. last point of the learning curve). Show the best performing model in bold, and compare it with each of 
#  the remaining models for statistical significance (see the slides from that module). Use two-tailed paired t-test 
#  from scipy (scipy.stats.ttest_rel). Those who are statistically significantly different from the best model, mark 
#  them with a * in the table.
# 6. In a separate table, compare the computational training and test times of all the models for the last point of the learning curve.

# Submission: 
#  1. [3 points] The Python program .py file in which you wrote your program (do not submit work saved from the Python prompt). 
#  The program should include loading the data step as well. The user should be able to run the program and generate the results by 
#  importing the file. If the dataset is not available online then also submit the dataset.
# 2. A short report (pdf, doc or docx file) that includes:
#   [1 points] A brief description of the dataset (what is the task, what are the features and the target)
#   [1 point] Describe the settings of the methods (e.g. whether there were missing feature values, what 
#   parameter values were searched for tuning, what percentages of training data were used for learning curves etc.)
#   [3 points] Results - In the form of graph and tables as described above.
#   [2 point] Discussion - Your thoughts and conclusions regarding (a few sentences for each):
#       - Learning curve comparison
#       - rmse comparison with full training data
#       - Computational time comparison

from cProfile import label
from statistics import mean
from sklearn import datasets
from sklearn.model_selection import cross_validate 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import learning_curve
import pandas as pd 
import scipy
from matplotlib import pyplot

lis = datasets.fetch_openml(data_id = 41928)

# Fetch the Dataaset Features 
#print("\n")
#print(lis.data.info())

# Transforming Nominal Features 
ct = ColumnTransformer([("encoder",OneHotEncoder(sparse = False),[22])],remainder = "passthrough")
new_data = ct.fit_transform(lis.data)
type(new_data)
lis_new_data = pd.DataFrame(new_data,columns = ct.get_feature_names_out(),index = lis.data.index)

# Fetch The Transformed Nominal Features Dataset
#print("\n")
#print(lis_new_data.info())

# DecisionTreeRegressor 
dtr = DecisionTreeRegressor(min_samples_leaf = 1)
dtr_scores = cross_validate(dtr,lis_new_data,lis.target,cv = 10,scoring = "neg_root_mean_squared_error")
print("\nDecisionTreeRegressor")
dtr_rmse = 0 - dtr_scores["test_score"]
dtr_rmse_mean = dtr_rmse.mean()
#print("dtr_rmse = ",dtr_rmse)
print("dtr_rmse_mean = ",dtr_rmse_mean)
dtr_train_sizes,dtr_train_scores,dtr_test_scores,dtr_fit_times,dtr_score_times = learning_curve(dtr,lis_new_data,lis.target,train_sizes = [0.2,0.4,0.6,0.8,1],cv = 10,return_times = True,scoring = "neg_root_mean_squared_error",shuffle = True,random_state = 0)
#print("dtr_train_scores = ",dtr_train_scores)
print("dtr_train_scores_mean = ",dtr_test_scores.mean())
#print("dtr_fit_times = ",dtr_fit_times)
print("dtr_fit_times_mean = ",dtr_fit_times.mean())
#print("dtr_score_times",dtr_score_times)
print("dtr_score_times_mean = ",dtr_score_times.mean())
lc_dtr_rmse = 0 - dtr_test_scores
#print("lc_dtr_rmse = ",lc_dtr_rmse)
print("lc_dtr_rmse_mean = ",lc_dtr_rmse.mean())
#print("dtr_train_sizes = ",dtr_train_sizes)
pyplot.plot(dtr_train_sizes,lc_dtr_rmse.mean(axis = 1),label = "DecisionTreeRegressor")

# KNearNeighborsRegressor 
knnr = KNeighborsRegressor(n_neighbors = 5)
knnr_scores = cross_validate(knnr,lis_new_data,lis.target,cv = 10,scoring = "neg_root_mean_squared_error")
print("\nKNearNeighborsRegressor:")
knnr_rmse = 0 - knnr_scores["test_score"]
knnr_rmse_mean = knnr_rmse.mean()
#print("knnr_rmse = ",knnr_rmse,)
print("knnr_rmse_mean = ",knnr_rmse_mean)
knnr_train_sizes,knnr_train_scores,knnr_test_scores,knnr_fit_times,knnr_score_times = learning_curve(knnr,lis_new_data,lis.target,train_sizes = [0.2,0.4,0.6,0.8,1],cv = 10,return_times = True,scoring = "neg_root_mean_squared_error",shuffle = True,random_state = 0)
#print("knnr_train_scores = ",knnr_train_scores)
print("knnr_train_scores_mean = ",knnr_test_scores.mean())
#print("knnr_fit_times = ",knnr_fit_times)
print("knnr_fit_times_mean = ",knnr_fit_times.mean())
#print("knnr_score_times",knnr_score_times)
print("knnr_score_times_mean = ",knnr_score_times.mean())
lc_knnr_rmse = 0 - knnr_test_scores
#print("lc_knnr_rmse = ",lc_knnr_rmse)
print("lc_knnr_rmse_mean = ",lc_knnr_rmse.mean())
#print("knnr_train_sizes = ",knnr_train_sizes)
pyplot.plot(knnr_train_sizes,lc_knnr_rmse.mean(axis = 1),label = "KNearNeighborsRegressor")

# LinearRegression 
lr = LinearRegression()
lr_scores = cross_validate(lr,lis_new_data,lis.target,cv = 10,scoring = "neg_root_mean_squared_error")
print("\nLinearRegression:")
lr_rmse = 0 - lr_scores["test_score"]
lr_rmse_mean = lr_rmse.mean()
#print("lr_rmse = ",lr_rmse)
print("lr_rmse_mean = ",lr_rmse_mean)
lr_train_sizes,lr_train_scores,lr_test_scores,lr_fit_times,lr_score_times = learning_curve(lr,lis_new_data,lis.target,train_sizes = [0.2,0.4,0.6,0.8,1],cv = 10,return_times = True,scoring = "neg_root_mean_squared_error",shuffle = True,random_state = 0)
#print("lr_train_scores = ",lr_train_scores)
print("lr_train_scores_mean = ",lr_test_scores.mean())
#print("lr_fit_times = ",lr_fit_times)
print("lr_fit_times_mean = ",lr_fit_times.mean())
#print("lr_scores_times",lr_score_times)
print("lr_score_times_mean = ",lr_score_times.mean())
lc_lr_rmse = 0 - lr_test_scores
#print("lc_lr_rmse = ",lc_lr_rmse)
print("lc_lr_rmse_mean = ",lc_lr_rmse.mean())
#print("lr_train_sizes = ",lr_train_sizes)
pyplot.plot(lr_train_sizes,lc_lr_rmse.mean(axis = 1),label = "LinearRegression")

# Support Vector Machine regressor: SVR in sklearn.svm 
svr = SVR()
svr_scores = cross_validate(svr,lis_new_data,lis.target,cv = 10,scoring = "neg_root_mean_squared_error")
print("\nSVR:")
svr_rmse = 0 - svr_scores["test_score"]
svr_rmse_mean = svr_rmse.mean()
#print("svr_rmse = ",svr_rmse)
print("svr_rmse_mean = ",svr_rmse_mean)
svr_train_sizes,svr_train_scores,svr_test_scores,svr_fit_times,svr_score_times = learning_curve(svr,lis_new_data,lis.target,train_sizes = [0.2,0.4,0.6,0.8,1],cv = 10,return_times = True,scoring = "neg_root_mean_squared_error",shuffle = True,random_state = 0)
#print("svr_train_scores = ",svr_train_scores)
print("svr_train_scores_mean = ",svr_test_scores.mean())
#print("svr_fit_times = ",svr_fit_times)
print("svr_fit_times_mean = ",svr_fit_times.mean())
#print("svr_scores_times",svr_score_times)
print("svr_score_times_mean = ",svr_score_times.mean())
lc_svr_rmse = 0 - svr_test_scores
#print("lc_svr_rmse = ",lc_svr_rmse)
print("lc_svr_rmse_mean = ",lc_svr_rmse.mean())
#print("svr_train_sizes = ",svr_train_sizes)
pyplot.plot(svr_train_sizes,lc_svr_rmse.mean(axis = 1),label = "SVR")

# Bagged decision tree regressor: BaggedRegressor 
bdtr = BaggingRegressor()
bdtr_scores = cross_validate(bdtr,lis_new_data,lis.target,cv = 10,scoring = "neg_root_mean_squared_error")
print("\nBaggedRegressor:")
bdtr_rmse = 0 - bdtr_scores["test_score"]
bdtr_rmse_mean = bdtr_rmse.mean()
#print("bdtr_rmse = ",bdtr_rmse)
print("bdtrr_rmse_mean = ",bdtr_rmse_mean)
bdtr_train_sizes,bdtr_train_scores,bdtr_test_scores,bdtr_fit_times,bdtr_score_times = learning_curve(bdtr,lis_new_data,lis.target,train_sizes = [0.2,0.4,0.6,0.8,1],cv = 10,return_times = True,scoring = "neg_root_mean_squared_error",shuffle = True,random_state = 0)
#print("bdtr_train_scores = ",bdtr_train_scores)
print("bdtr_train_scores_mean = ",bdtr_test_scores.mean())
#print("bdtr_fit_times = ",bdtr_fit_times)
print("bdtr_fit_times_mean = ",bdtr_fit_times.mean())
#print("bdtr_scores_times",bdtr_score_times)
print("bdtr_score_times_mean = ",bdtr_score_times.mean())
lc_bdtr_rmse = 0 - bdtr_test_scores
#print("lc_bdtr_rmse = ",lc_bdtr_rmse)
print("lc_bdtr_rmse_mean = ",lc_bdtr_rmse.mean())
#print("bdtr_train_sizes = ",bdtr_train_sizes)
pyplot.plot(bdtr_train_sizes,lc_bdtr_rmse.mean(axis = 1),label = "BaggedRegressor")

# DummyRegressor 
dr = DummyRegressor()
dr_scores = cross_validate(dr,lis_new_data,lis.target,cv = 10,scoring = "neg_root_mean_squared_error")
print("\nDummyRegressor:")
dr_rmse = 0 - dr_scores["test_score"]
dr_rmse_mean = dr_rmse.mean()
#print("dr_rmse = ",dr_rmse)
print("dr_rmse_mean = ",dr_rmse_mean)
dr_train_sizes,dr_train_scores,dr_test_scores,dr_fit_times,dr_score_times = learning_curve(dr,lis_new_data,lis.target,train_sizes = [0.2,0.4,0.6,0.8,1],cv = 10,return_times = True,scoring = "neg_root_mean_squared_error",shuffle = True,random_state = 0)
#print("dr_train_scores = ",dr_train_scores)
print("dr_train_scores_mean = ",dr_test_scores.mean())
#print("dr_fit_times = ",dr_fit_times)
print("dr_fit_times_mean = ",dr_fit_times.mean())
#print("dr_scores_times",dr_score_times)
print("dr_score_times_mean = ",dr_score_times.mean())
lc_dr_rmse = 0 - dr_test_scores
#print("lc_dr_rmse = ",lc_dr_rmse)
print("lc_dr_rmse_mean = ",lc_dr_rmse.mean())
#print("dr_train_sizes = ",dr_train_sizes)
pyplot.plot(dr_train_sizes,lc_dr_rmse.mean(axis = 1),label = "DummyRegressor")

#Statistical Significance 
print("\n")
print("Comparing BaggedRegressor with DecisionTreeRegressor: ",scipy.stats.ttest_rel(lc_bdtr_rmse.mean(axis = 1),lc_dtr_rmse.mean(axis = 1)))
print("Comparing BaggedRegressor with KNearNeighborsRegressor: ",scipy.stats.ttest_rel(lc_bdtr_rmse.mean(axis = 1),lc_knnr_rmse.mean(axis = 1)))
print("Comparing BaggedRegressor with SVR: ",scipy.stats.ttest_rel(lc_bdtr_rmse.mean(axis = 1),lc_svr_rmse.mean(axis = 1)))
print("Comparing BaggedRegressor with LinearRegression: ",scipy.stats.ttest_rel(lc_bdtr_rmse.mean(axis = 1),lc_lr_rmse.mean(axis = 1)))
print("Comparing BaggedRegressor with DummyRegressor: ",scipy.stats.ttest_rel(lc_bdtr_rmse.mean(axis = 1),lc_dr_rmse.mean(axis = 1)))

print("\n")
pyplot.title("Learning Curves")
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.legend()
pyplot.show()