import time
from random import choice
from copy import deepcopy
import matplotlib.pyplot as plt

class SA_Directional:
    """
    Implementation of greedy directional simulated annealing (SA).
    
    Used for hyperparameter tuning of models with high evaluation time,
    for which traditional SA, random processes, and grid search are
    more or less intractable. 
    
    Given an estimator and the parameters you wish to optimize on (as
    well as their search domains), returns the parameter selection
    which resulted in greatest model performance.
    
    When a parameter selection results in improved model performance,
    this improvement is seized upon greedily; rather than selecting a
    new random paremeter after each improvement, the algorithm makes
    a random choice that is bounded in the direction that improvement
    occured (e.g., if increasing a parameter lead to a good outcome in
    iteration n, only random selections of that parameter in the
    positive direction will be allowed in iteration n+1). If the
    selection in iteration n+1 is rejected, the algorithm returns to
    standard simulated annealing procedure. In this way, the algorithm
    alternates between standard simulated annealing and greedy hill
    climbing.
    
    If the search domain of the variable selected for directional 
    alteration does not have clear directions, e.g., if it is nominal
    or is numerical but already at the limits of its range, the
    algorithm returns to simulated annealing procedure and makes a new
    random hyperparameter selection.
    
    Basis for simulated annealing comes from description given by
    https://santhoshhari.github.io/simulated_annealing/
    
    -------------------------------------------------------------
    
    Example usage:
        
    params = {
        'n_trees':list(range(75,226)),
        'criterion':('gini','entropy'),
        'min_samples_split':[i/1000 for i in range(1,501,1)],
        'min_samples_leaf':[i/1000 for i in range(1,501,1)],
        'max_features':[0.1,0.5,1.0,'sqrt','log2'],
        'min_impurity_decrease':[i/1000 for i in range(0,501,1)],
        'k':[i/10000 for i in range(1,1000,1)]
        }

    sa = SA_Directional(RandomForest_knn, params, temp=1.0, temp_velocity=0.975)
    sa.fit(X_train, X_test, y_train, y_test, rejection_streak_limit=10)
    -------------------------------------------------------------
    """
    def __init__(self, Estimator, params, temp=1.0, temp_velocity=.9, 
                 beta=1, temp_cutoff=0.00001):
        """
        param Estimator: a model class with a fit and score function from which 
            to instantiate models
        param params: dict,
            maps model's parameters to their search domain.
        param init_temp: int or float,
            initial temperature
        param temp_velocity: float [0,1.0],
            factor by which temperature scales down after temp_iter iterations
        param temp: int,
            number of iterations after which temperature is altered
        param beta: int or float,
            normalizing constant. Increasing beta results in higher
            probability of rejecting candidates.
        """
        self.Estimator = Estimator
        self.params = params
        self.temp = temp
        self.temp_velocity = temp_velocity
        self.beta = beta
        self.temp_cutoff = temp_cutoff
        
        self.best_estimator = None
        
    
    def isAccepted(self, current_eval, candidate_eval)->bool:
        """
        Assumes maximization: higher score is better.
        
        param current_eval: last accepted model iteration
        param candidate_eval: candidate model's evaluation score
        """
        p_accept = np.exp(self.beta*(candidate_eval-current_eval)/(self.temp))
        
        print()
        print("Current_eval: {}; Candidate_eval: {}".format(current_eval, candidate_eval))
        print("Current temperature: {}".format(self.temp))
        print("Acceptance probability: {}".format(p_accept))
        print()
        self.temp = self.temp*self.temp_velocity
        
        return p_accept > np.random.uniform(low=0.0, high=1.0)
    
    
    def isNumeric(self, arr)->bool:
        """
        arr: list of possible values for a parameter
        returns True if all elems of array are numeric
        """
        return np.all(np.array([(type(i)==int or type(i)==float) for i in arr]))
    
    
    def getNewDomain(self, arr, comparator)->np.array:
        """
        Checks if arr is numeric. If it is, returns subset of arr satisfying comparator (may be empty if
        old_value at extremes of the range of arr). If not, return empty array.
        """
        if self.isNumeric(arr):
            new_domain = np.array([i for i in arr if comparator(i)])
        else:
            new_domain = np.array([])
            
        return new_domain
    
    
    def updateRandomly(self, current_params:dict, history:list, avoid_variable=None, loop_limit=100):
        """
        Returns a random parameter set with one updated value that has not been seen in the
        history.
        Raises an exception of loop_limit-many updates are attempted.
        
        param current_params: dictionary containing parameters for model
        param history: history of past parameter maps
        param avoid_variable: variable to avoid selecting.
        param loop_limit: update limit for selecting new states.
        """
        choices = list(current_params.keys())
        
        if avoid_variable:
            choices.remove(avoid_variable)
        
        candidate_params = deepcopy(current_params)
        reassignments = 0
        
        while candidate_params in history and reassignments < loop_limit:
            candidate_params = deepcopy(current_params)
            new_variable = choice(choices)
            old_value = current_params[new_variable]
            candidate_params[new_variable] = choice(self.params[new_variable])
            reassignments+=1
            
        if reassignments >= loop_limit:
            raise Exception("loop limit exhausted during state update phase.")
            
        return (new_variable, old_value, candidate_params)
        
        
    def printCandidateState(self, current_params, candidate_eval, new_variable="No modification")->None:
        """
        param current_state: dict
        param candidate_eval: current evaluation score of candidate
        """
        print(current_params)
        print("Modified variable: {}".format(new_variable))
        print("Above model evaluated with score of {}.".format(candidate_eval))
        print()
        return None
    
    
    def fit(self, X_train, X_test, y_train, y_test, rejection_streak_limit=3,
            max_runtime=21600):
        """
        Applies directional SA algorithm as follows:
        
        1. Randomly initialize all hyperparameters (hp's), treat this as 
        current state, and evaluate model performance.
        
        2. Either randomly select one hp and randomly update its value,
        or, if the previous iteration resulted in better performance,
        select the previously altered variable and increment it
        randomly in the direction of previous change. (If the latter is
        not possible, return to standard procedure and make a new
        random hp selection).
        
        3. If the combination has already been visited, return to step 2.
        
        4. Evaluate estimator performance on the state derived in 2.
        
        5. Accept or reject new state based on product of 
        temperature conditions and performance difference from
        previous state.
        
        6. Repeat 2 though 5 until max_runtime reached or until
        no model is accepted after rejection_streak_limit many
        rejections are made
        
        ------------------------------------------------------------------
        
        param X_train: iterable of type list, np.array, sparse matrix, etc.
        param y_train: iterable of type list, np.array, sparse matrix, etc.
        param X_test: iterable of type list, np.array, sparse matrix, etc.
        param y_test: iterable of type list, np.array, sparse matrix, etc.
        param rejection_streak_limit: int,
            end tuning and return current state after this many 
            candidate rejections in a row have occured.
        param max_runtime: int, default 21600 seconds (6 hours),
            time (in seconds) after which fitting will end and the current
            best model will be returned.
        
        returns best estimator found via directional SA
        """
        start_time = time.time()        
        
        print("Beginning execution at {} epoch time".format(start_time))
        
        param_history = []
        current_params = {k:choice(v) for k,v in self.params.items()}
        
        current_state = self.Estimator(**current_params)
        current_state.fit(X_train, y_train)
        current_eval = current_state.score(X_test, y_test)
        
        self.printCandidateState(current_params, current_eval)
        
        param_history.append(current_params)
            
        rejection_streak = 0
        
        # make first random variable selection
        new_variable, old_value, current_params = self.updateRandomly(current_params, param_history)
        
        current_time = time.time()
        
        while (rejection_streak < rejection_streak_limit) and (current_time - start_time < max_runtime) and (self.temp > self.temp_cutoff):
            
            candidate_state = self.Estimator(**current_params)
            candidate_state.fit(X_train, y_train)
            candidate_eval = candidate_state.score(X_test, y_test)
            
            param_history.append(current_params)
            
            self.printCandidateState(current_params, candidate_eval, new_variable)
                
            is_improvement = candidate_eval > current_eval
            accepted = self.isAccepted(current_eval, candidate_eval)
            
            if accepted:
                print("State accepted")
                print()
                current_state = candidate_state
                current_eval = candidate_eval
                rejection_streak = -1
            else:
                print("State rejected")
                print()
            
            if is_improvement and accepted:
                if current_params[new_variable] > old_value:
                    domain = self.getNewDomain(self.params[new_variable], lambda x: x > current_params[new_variable])
                else:
                    domain = self.getNewDomain(self.params[new_variable], lambda x: x < current_params[new_variable])
                    
                if domain.size == 0:
                    new_variable, old_value, current_params = self.updateRandomly(current_params,
                                                                                  param_history,
                                                                                   avoid_variable=new_variable)
                else:
                    current_params[new_variable] = choice(domain)
                        
            else:
                print("no improvement or rejected")
                new_variable, old_value, current_params = self.updateRandomly(current_params,
                                                                                     param_history)
            
              
            
            rejection_streak += 1
            current_time = time.time()
            
            print("-"*30)
            
        self.best_estimator = current_state
        
        
        return current_state, current_eval
            
        
class RandomTuner:
    """
    A true random walker: trains and evaluates random models until a 
    given time limit is reached. Cross-validition is not used, as
    it is assumed that the given Estimator has a prohibitively long
    evaluation time. By saving the search history, RandomTuner is
    meant to provide a rough heuristic of the effect different
    hyperparemeters have on model performance. Although
    many models may have several variables, making tuning a 
    multi-dimensional problem, RandomTuner provides functions for
    plotting model performance as specific parameters are changed.
    Though this does not provide insight into specific global
    maximum combinations of variables, it may help in limiting
    the neighborhood of reasonable search later on, especially
    if traditional means of searching the hyperparameter space
    are intractible due to model complexity.
    """
    def __init__(self, Estimator, params):
        """
        param Estimator: a model class with a fit and score function
        param params: dict,
            maps model's parameters to their search domain.
        """
        self.Estimator = Estimator
        self.params = params
        
        self.best_estimator = None
        self.search_history = []
        
        
    def fit(self, X_train, X_test, y_train, y_test, max_runtime=18000, verbose=True):
        """
        Randomly searches space, generating and testing models.
        Returns best model found after search; also saves this and the
        complete history of tested models to best_estimator and
        search_history attributes, respectively.
        
        param X_train: iterable of type list, np.array, sparse matrix, etc.
        param X_test: iterable of type list, np.array, sparse matrix, etc.
        param y_train: iterable of type list, np.array, sparse matrix, etc.
        param y_test: iterable of type list, np.array, sparse matrix, etc.
        param max_runtime: int, default 18000 seconds (6 hours),
            approximate length of time to run search, 
        param verbose: bool, default True
        
        returns best estimator (maximum accuracy)
        """
        start_time = time.time()
        
        while(time.time() - start_time < max_runtime):
            
            this_clf_time = time.time()
            
            current_params = {key:choice(v) for k,v in self.params}
            
            this_clf = Estimator(**current_params)
            
            print("Evaluating params for {}".format(current_params))
            
            this_clf.fit(X_train, y_train) 
            acc = this_clf.score(X_test, y_test)
            
            this_clf_time = time.time() - this_clf_time
            
            self.search_history.append((acc,current_params,this_clf_time))
            
            print("Scored: {}".format(acc))

            print("time elapsed: {}".format(this_clf_time))

            print("Max score so far: {}".format(max(accuracies)))
            print("+-=+-=+-=+-=Completed testing for this clf+-=+-=+-=")
            print()
            
        self.best_estimator = max(self.search_history)
        
        return self.best_estimator
    
    
    def chartReport(self, plot_vs_acc=True, plot_vs_time=False)->None:
        """
        Produces graphs for model accuracy and evaluation time 
        versus corresponding values for each of the tested 
        hyperparameters.
        
        param plot_vs_acc: bool, default True,
            If True, plots graphs for total model performance versus
            tested hyperparameter values for each search parameter.
            
        param plot_vs_time: bool, default False,
            If True, plots graphs for model evaluation time versus
            tested hyperparameter values for each search parameter.
        """
        param_values = {k:[] for k in self.search_history[0][1]}
        
        for model in self.search_history:
            for param in model[1]:
                param_values[param].append(model[1][param])
        
        if plot_vs_acc:
            scores = [i[0] for i in self.search_history]
            for param, values in param_values.items():
                plt.scatter(values, scores)
                plt.title("{} vs. accuracy on randomly instantiated models".format(param))
                plt.xlabel(param)
                plt.ylabel("model accuracy")
                plt.show()
                
        if plot_vs_time:
            times = [i[2] for i in self.search_history]
            for param, values in param_values.items():
                plt.scatter(values, times)
                plt.title("{} vs. evaluation time on randomly instantiated models".format(param))
                plt.xlabel(param)
                plt.ylabel("model evaluation time")
                plt.show()
                
                
        return None