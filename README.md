# GAHS
Genetic Algorithm Hybrid Stacking

The GAHS is a multi-porpuse optimazation methodology resulting to the best combination of base-learners, features and meta-learners for ensemble stacking. The procedure is expecting a pandas dataframe containing i) estimations of various machine learning models (base-learners), ii) a set of the initial  features used to make predictions with the base-learners, iii) the true values of the target variable for training phase and iv) a collumn defining the k-fold cross validation splits. Additionally, a list of the meta-learners should be defined. (it is highly recommended using "fast" or GPU accelerated algorithms). The final aoutcome is the optimal configuration of ensemble stacking. 

It is also recommended training various base-learners (e.g. tree based, ANNs, SVMs with numerous combinations of hyperparameters) in different subsets of features, derived from feature selection methods (e.g. Random Forest Feature Importance, metaheuristics, Correlation based Feature Selection).

# Example of usage
```python
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

run = 'OBSERVED'

df = pd.read_csv()

target = df[run]

X = df.drop([run], axis=1)

LRmodel = LinearRegression()

KNN = KNeighborsRegressor(n_neighbors=5, n_jobs =-1)

XGB50 = XGBRegressor(n_estimators = 50, 
                                colsample_bytree=.8,
                                learning_rate=0.1,
                                alpha=.1,
                                tree_method='gpu_hist',
                                subsample=.8, objective='reg:squarederror')
                                
XGB300 = XGBRegressor(n_estimators = 300, 
                                colsample_bytree=.8,
                                learning_rate=0.1,
                                alpha=.1,
                                tree_method='gpu_hist',
                                subsample=.8, objective='reg:squarederror'
                                )

X = df.drop([run, 'Fold'], axis=1)

In_models = np.array([LRmodel, mean, XGB50, XGB300, KNN]) 

chromo, score = genetic_algorithm(n_gen=500, size=23, n_feat=len(X.columns),
                                  models=In_models, fitness_function='mse',
                                  selection_constant=3, crossover_prob=0.7, mutation_prob=0.05,
                                  df=df, target=run, category='Fold')

pd.Series(score).plot()

print('Chosen base-learners and features',X.columns[chromo[-1][:-len(In_models)]])

print('Chosen meta-learner',In_models[chromo[-1][-len(In_models):]])
```

