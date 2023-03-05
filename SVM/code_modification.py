import numpy as np
import pandas as pd
import time
from itertools import permutations

##  This function will help us to create automatically the transforms list of lists.
def unique_permutations(iterable, r=None):
    '''
    Creates a list of list of all (r) permutations of a  set of elements (iterable).

    Parameters
    ----------
    iterable : any type of Python iterable as list, tuple or string
        Stores the elements of the permutation

    r : Int
        Number of elements to use in the permutation     
    
    Returns
    ----------
    transforms : list
        Permutations 
    '''
    previous = tuple()
    transforms = []
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            transforms.append(list(p))
    
    return transforms  

class Support_Vector_Machine:
    """
    Support Vecto Machine model.

    ...

    Attributes
    ----------
    b_range_multiple : int 
            Parameter to help optimize the value of b
            
    b_multiple : int 
        Parameter to help optimize the value of b   

    Methods
    -------
    fit(X, y):
        Fit the Support Vector Machine Model
    
    predict(features):
        Predict using the Support Vector Machine Model
    
    """
    
    def __init__(self, b_range_multiple = 5, b_multiple = 5):            
        '''
        Initializes an object of the class of Support Vector Machine.

        Parameters
        ----------
        b_range_multiple : int 
            Parameter to help optimize the value of b
            
        b_multiple : int 
            Parameter to help optimize the value of b       
        '''
        
        self.b_range_multiple = b_range_multiple
        self.b_multiple = b_multiple        
    
    def fit(self, X, y):
        '''
        Fit the Support Vector Machine Model

        Parameters
        ----------
        X : (array-like, matrix) of shape (n_samples, n_features)
            Training data
            
        y : (array-like, matrix) of shape (n_features, 1) or (n_samples, n_targets)
            Classes of the samples           
        '''
        
        self.data = X
        self.classes = y
        
        ## In this dictionary (opt_dict), we store the norm of the vector w, along with the values of w and b.
        ## At the end of the iterations, we will choose the lowest norm of all the evaluated w vectors,
        ## from this we will choose our optimized parameters.
        opt_dict = {}
        
        transforms = unique_permutations(self.data.shape[1]*[1] + self.data.shape[1]*[-1], self.data.shape[1])
        
        ## Since our data X is a numpy array, we can find the max (min) of it using the function np.amax (np.amin),
        ## meaning theres is no need to iterate over all the array and create a new list "all_data" for this.
        
        # all_data = []
        # for yi in self.data:
        #     for featureset in self.data[yi]:
        #         for feature in featureset:
        #             all_data.append(feature)

        self.max_feature_value = np.amax(self.data)
        self.min_feature_value = np.amin(self.data)
        # all_data = None            

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,                      
                      self.max_feature_value * 0.001,]
        
        
        b_range_multiple = self.b_range_multiple
        b_multiple = self.b_multiple
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.full((self.data.shape[1],), latest_optimum)    
            print(w)      
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True                        
                        
                        ## Since our data X and classes y are numpy arrays, we can avoid having two nested for statements.
                        ## Also, if for one x_i the condition of yi*(np.dot(w_t,xi)+b) >= 1 is not satisfied,
                        ## we should stop the verification for the rest of samples, that's why we added a break 
                        ## statement after the "found_option = False" line.
                        
                        # for i in self.data:
                        #     for xi in self.data[i]:
                        #         yi=i
                        #         # Verifiy constraints
                        #         if not yi*(np.dot(w_t,xi)+b) >= 1:
                        #             found_option = False
                        
                        for i in range(len(self.data)):
                            yi = self.classes[i]
                            xi = self.data[i]
                                # Verifiy constraints
                            if not yi*(np.dot(w_t,xi)+b) >= 1:
                                found_option = False
                                break
                                    
                        if found_option:
                            # Computes norm
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    print(w, 'hi')
                    optimized = True
                    ## Since we will be working with large datasets, we will avoid printing the following line
                    print('Optimized a step.')
                else:
                    ## Here we update the value of w with kind of gradient descent algorithm.                    
                    w = w - step
            if len(opt_dict)>0:
                norms = sorted([n for n in opt_dict])
                #||w|| : [w,b]
                print(norms, opt_dict, step, step_sizes)
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0]+step*2
            else:
                self.w = w
                self.b = b
                latest_optimum = w+step*2
        
            

    def predict(self, features):
        '''
        Predict using the Support Vector Machine Model

        Parameters
        ----------
        X : (array-like, matrix) of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions
        Returns
        ----------
        y_pred : (ndarray) of shape (n_features,)
            Predictions for X         
        '''
        # sign( x.w+b )
        y_pred = np.sign(np.dot(features, self.w)+self.b)
        return y_pred
    
df1 = pd.read_csv("data/banknote_authentication/data_banknote_authentication.txt", sep=",", header=None, names=["variance", "skewness", "curtosis", 
                                                                                        "entropy", "class"])
df1 = df1.replace({'class': 0}, {'class': -1})

X_1 = df1.copy().iloc[:, :-1].values # All data samples attributes
y_1 = df1.copy().iloc[:, -1].values # All data samples classes

svm1 = Support_Vector_Machine() # Create model

start_time = time.time()
svm1.fit(X_1, y_1) # Train model
print("--- %s seconds ---" % (time.time() - start_time))


y_1_pred = svm1.predict(X_1) 