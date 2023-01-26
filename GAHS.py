import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score, mean_squared_error
import random
from tqdm import tqdm
import os
from copy import deepcopy



def initilization_of_population(size, n_feat, models):
    '''
    size:   population size
    n_feat: number of features
    models: list of meta learners e.g. ['mean', LinearRgression(), ...]
    RETURNS:list with a randomly initialized population of chromosomes
    '''
    population = []
    for i in range(size):
        # initialize the first part of chromosome containing base-learnears' predictions and initial features
        chromosome1 = np.ones(n_feat, dtype=np.bool)
        per = random.uniform(0.1, 0.9)
        chromosome1[int(per * n_feat):] = False
        np.random.shuffle(chromosome1)

        # initialize =the second part of the chromosome containing the meta-learners
        chromosome2 = np.ones(len(models), dtype=np.bool)
        chromosome2[:-1] = False
        np.random.shuffle(chromosome2)

        # combine base-learners' predictions, features and models to chromosome = [base-learners, features, models]
        chromosome = np.hstack([chromosome1, chromosome2])

        population.append(chromosome)

    return population


def evaluation(population_for_eval, models, fitness_function, df, target, category):
    '''
    population_for_eval: list of chromosomes
    models:              list of meta learners'
    fitness_function:    options "r2" or "mse"
    df:                  pandas dataframe || should contain i) base-learners' estimations, ii) initial features or a subset of them, iii) the true values
    or target variable and iv) a column named category defining the fold split (e.g. if it is a spatiotemporal problem either a category defining the 
    different locations or an identifier for the time dimension)
    target:              string, should be included in df, name od target variable column
    category:            string, variable that splits tha data for k-fold cross validatation, should be included in df
    RETURNS:             2 sorted lists with the scores and the chromosomes
    '''

    names = df[category].unique().tolist()
    Y = df[target]
    X = df.drop([target, category], axis=1)

    scores = []
    population = deepcopy(population_for_eval) 

    for chromosome in tqdm(population):  

        all_preds, true = [], []

        models_place = chromosome[-len(models):]
        s_models = models[models_place]

        features = chromosome[:-len(models)]

        if len(s_models) == 1:  # fail-safe senario, in case of more than two meta-learners are chooses (it could happen in the crossover)
            ch_model = s_models[0]

            for name in names:

                Test_indx = df[df[category] == name].index
                X_train = X.iloc[~df.index.isin(Test_indx), :]
                y_train = Y.iloc[~df.index.isin(Test_indx)]
                # y_train = np.log1p(y_train) # for logarithmic transformation
                X_test = X.iloc[Test_indx, :]
                y_test = Y.iloc[Test_indx]

                if ch_model == 'mean':
                    predictions = X_test.iloc[:, features].mean(axis=1)

                elif ch_model == 'MLR':
                    model = LinearRegression()#LassoLars(0.01)
                    model.fit(X_train.iloc[:, features], y_train)
                    predictions = model.predict(X_test.iloc[:, features])

                elif ch_model == 'XGB':
                    model = XGBRegressor(n_estimators = 100, 
                                #colsample_bytree=.8,
                                learning_rate=0.15,
                                #alpha=.1,
                                #tree_method='gpu_hist',
                                #subsample=.8, 
                                         n_jobs=-1,
                                objective='reg:squarederror')

                    model.fit(X_train.iloc[:, features], y_train)
                    predictions = model.predict(X_test.iloc[:, features])
                elif ch_model == 'LGB':
                    model = LGBMRegressor(n_estimators=100, learning_rate=0.15, n_jobs=-1, verbose=-1)
                    model.fit(X_train.iloc[:, features], y_train)
                    predictions = model.predict(X_test.iloc[:, features])

                all_preds.append(predictions)
                true.append(y_test)

            # all_preds = np.expm1(np.concatenate(all_preds)) # for logarithmic transformation
            all_preds = np.concatenate(all_preds)
            true = np.concatenate(true)
            
            if fitness_function == 'r2':
                score = r2_score(true, all_preds)
            elif fitness_function == 'mse':
                score = mean_squared_error(true, all_preds)

        else:  # if more than one model exists in the chromosome assing a bad score
            if fitness_function == 'r2':
                score = -9999.0
            elif fitness_function == 'mse':
                score = 9999.0

        scores.append(score)

    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)

    if fitness_function == 'r2':
        return list(scores[inds][::-1]), list(population[inds][::-1])
    elif fitness_function == 'mse':
        return list(scores[inds]), list(population[inds])


def selection_tournament(pop_after_fit_in, scores, selection_constant):
    '''
    pop_after_fit_in:   sorted list of chromosomes after evaluation
    scores:             sorted list of scores after evaluation
    selection_constant: number of chromosomes to compete
    RETURNS:            the selected next generation of chromosomes
    '''

    # elitism, keep the best chromosome twice
    pop_after_fit = deepcopy(pop_after_fit_in)
    population_nextgen = [pop_after_fit[0], pop_after_fit[0]]

    # perfom tournament selection for the rest of the population
    for i in range(2, len(pop_after_fit)):
        index2 = []
        for j in range(0, selection_constant):
            index2.append(random.randint(0, len(pop_after_fit) - 1))

        index3 = np.amin(index2)
        population_nextgen.append(pop_after_fit[index3])

    return population_nextgen


def crossover(pop_after_sel_in, crossover_prob, models):
    '''
    pop_after_sel_in: the selected next generation
    crossover_prob:   propability of individuals mating
    RETURNS:          the population after crossover
    '''
    pop_after_sel = deepcopy(pop_after_sel_in)
    pop_after_cross = [pop_after_sel[0]]


    j = 0
    for i in range(1, len(pop_after_sel)):
        rnd = random.random()
        if rnd < crossover_prob:

            j = j + 1
            if j == 1:
                father = pop_after_sel[i]
            else:

                mother = pop_after_sel[i]
                j = 0

                # 1 point crossover to create two children
                crpoint = int(random.random() * (len(pop_after_sel[0]) - len(models)))  

                chromo1 = father[:crpoint]
                chromo2 = mother[crpoint:]
                chromo3 = mother[:crpoint]
                chromo4 = father[crpoint:]
                child1 = np.concatenate((chromo1, chromo2), axis=0)
                child2 = np.concatenate((chromo3, chromo4), axis=0)


                pop_after_cross.append(child1)
                pop_after_cross.append(child2)

        else:
            pop_after_cross.append(pop_after_sel[i])
        if i == len(pop_after_sel) - 1 and len(pop_after_sel) - len(pop_after_cross) == 1: pop_after_cross.append(
            pop_after_sel[i])

    return pop_after_cross


def mutation(pop_after_cr, mutation_prob, models):
    '''
    pop_after_cr:  list of the output population from crossover
    mutation_prob: propability of a gene to mutate
    models:        list of models, used only in len()
    returns:       the mutated next generation
    '''
    # elitism, best chromosome does not pass mutation, only it's copy
    population_mut = [pop_after_cr[0]]


    new_pop = deepcopy(pop_after_cr)
    for i in range(1, len(new_pop)):

        chromosome = new_pop[i]

        # mutate each gene (feature) given the probability
        for j in range(len(chromosome[:-len(models)])):
            if random.random() < mutation_prob:
                chromosome[j] = not chromosome[j]


        # given the probability mutate the model part of the chromosome as a whole to keep one model
        for j in range(len(chromosome[-len(models):])):
            if random.random() < mutation_prob:
                chromosome[-len(models):] = np.array([False] * len(models))
                ind = random.randint(len(chromosome) - len(models), len(chromosome) - 1)

                chromosome[ind] = True
                break

        population_mut.append(chromosome)

    return population_mut


def GAHS(size, n_feat, models, fitness_function, selection_constant, crossover_prob, mutation_prob, n_gen,
                      df, target, category):
    '''
    size: int, population size
    n_feat: int, number of features + base learners' estimations
    models: np.array, list of meta-learners' models
    fitness_function: string, mse or r2
    selection_constant: float, constant for tournament selection (recommended 2-5)
    crossover_prob: float, probability for a chromosome to pass through crossover [0 - 1.0] (recommended 0.5-0.8)
    mutation_prob: float, probability for a gene to pass through mutation [0 - 1.0] (recommended ?)
    n_gen: int, number of generations
    df: pandas dataframe || should contain i) base-learners' estimations, ii) initial features or a subset of them, iii) the true values
    or target variable and iv) a column named category defining the fold split (e.g. if it is a spatiotemporal problem either a category defining the 
    different locations or an identifier for the time dimension)
    target:              string, should be included in df, name od target variable column
    category:            string, variable that splits tha data for k-fold cross validatation, should be included in df
    Optimizer: Finds the best stacking combination of base-learners', features + the best meta-learner to fit them.
    RETURNS:   1) list of best features 2) list of best scores, for every generation
               the last elements of the lists are the optimal
    '''
    best_chromo, best_score = [], []

    in_pop = initilization_of_population(size, n_feat, models)

    for i in range(n_gen):
        scores, pop_fit = evaluation(in_pop, models, fitness_function, df, target, category)

        print(scores, i)

        pop_sel = selection_tournament(pop_fit, scores, selection_constant)

        pop_cross = crossover(pop_sel, crossover_prob, models)

        in_pop = mutation(pop_cross, mutation_prob, models)

        best_chromo.append(pop_fit[0])
        best_score.append(scores[0])

    return best_chromo, best_score

