# GAHS
Genetic Algorithm Hybrid Stacking

The GAHS is a multi-porpuse gptimazation methodology resulting to the best combination of base-learners, features and meta-learners for ensemble stacking. The procedure is expecting a pandas dataframe containing i) estimations of various machine learning models (base-learners), ii) a set of the initial  features used to make predictions with the base-learners, iii) the true values of the target variable for training phase and iv) a collumn defining the k-fold cross validation splits. Additionally, a list of the meta-learners should be defined. (it is highly recommended using "fast" or GPU accelerated algotritmhs)

