# GAHS
*Genetic Algorithm Hybrid Stacking*

The GAHS is a multi-purpose optimization methodology resulting in the best combination of base-learners, features and meta-learners for ensemble stacking. The procedure is expecting a pandas dataframe containing i) estimations of various machine learning models (base-learners), ii) a set of the initial features used to make predictions with the base-learners, iii) the true values of the target variable for the training phase and iv) a column defining the k-fold cross validation splits. Additionally, a list of meta-learners should be defined. (it is highly recommended to use "fast" or GPU accelerated algorithms). The final outcome is the optimal configuration of ensemble stacking. 

It is also recommended to train various base-learners (e.g. tree-based, ANNs, SVMs with numerous combinations of hyperparameters) in different subsets of features, derived from feature selection methods (e.g. Random Forest Feature Importance, metaheuristics, Correlation based Feature Selection).

# Data Sample

The [Beijing Multi-Site Air-Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data) from UCI Machine Learning Repository will be used for testing the GAHS procedure. In this case, we aim to depict the PM2.5 values in each monitoring station using input from all other monitoring stations.

# Example of usage

**First Step - Feature Engineering & Base Learners:**
Download the [data](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data) in a local directory. Then run the Base Learners.py. The final outcome should be a csv file (Hybrid_BaseLearners.csv), containing i) the hybrid dataset (features + base-learners), ii) the true values, and iii) a column defining the folds.

**Second Step - GAHS:** Apply following code with the use of GAHS.py

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

run = 'PM2.5'
# Download Sample Data to path

df = pd.read_csv('path/Hybrid_BaseLearners.csv ')

target = df[run]

X = df.drop([run], axis=1)

LRmodel = LinearRegression()

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

In_models = np.array([LRmodel, XGB50, XGB300]) 

chromo, score = GAHS(n_gen=500, size=23, n_feat=len(X.columns),
                                  models=In_models, fitness_function='mse',
                                  selection_constant=3, crossover_prob=0.7, mutation_prob=0.05,
                                  df=df, target=run, category='Fold')

pd.Series(score).plot()

print('Chosen base-learners and features',X.columns[chromo[-1][:-len(In_models)]])

print('Chosen meta-learner',In_models[chromo[-1][-len(In_models):]])
```
# Cite As
The paper introducing the GAHS methodology is under revision. Until then use this citation https://doi.org/10.3390/atmos13030416 presenting a part of the algorithm.
