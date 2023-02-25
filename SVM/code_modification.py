import numpy as np
import pandas as pd
import time

class Support_Vector_Machine:
    def __init__(self, b_range_multiple = 5, b_multiple = 5):            
        '''
        Initializes an object of the class of Support Vector Machine.

            Parameters:
                    b_range_multiple (int): ADD WHAT IS THIS
                    b_multiple (int): ADD WHAT IS THIS       
        '''
        
        self.b_range_multiple = b_range_multiple
        self.b_multiple = b_multiple
        
    # train
    def fit(self, X, y):
        '''
        Fit the Support Vector Machine Model.

            Parameters:
                    X (array-like, matrix) of shape (n_samples, n_features): Training data.
                    y (array-like, matrix) of shape (n_features, 1) or (n_samples, n_targets): Classes of the samples           
        '''
        
        self.data = X
        self.classes = y
        # { ||w||: [w,b] }
        # In this dictionary, we store the norm of the vector w, along with the values of w and b.
        # At the end of the iterations, we will choose the lowest norm of all the evaluated w vectors,
        # from this we will choose our optimized parameters
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        # all_data = []
        # for yi in self.data:
        #     for featureset in self.data[yi]:
        #         for feature in featureset:
        #             all_data.append(feature)

        self.max_feature_value = np.amax(self.data)
        self.min_feature_value = np.amin(self.data)
        # all_data = None

        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,]

        
        
        # extremely expensive
        b_range_multiple = self.b_range_multiple
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = self.b_multiple
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        
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
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(features,self.w)+self.b)
        return classification
        
        
data_dict = {"ft1": [1, 2, 3, 5, 6, 7], "ft2": [7, 8, 8, 1,-1,3], "y":[-1, -1, -1, 1, 1 ,1]}
    
df = pd.DataFrame.from_dict(data_dict)
X = df.copy().iloc[:, :-1].values
y = df.copy().iloc[:, -1].values


svm1 = Support_Vector_Machine()
t = time.time()
svm1.fit(X, y)
print(time.time() - t)

print(svm1.predict(np.array([[7, 3.5], [4, 1]])), svm1.w, svm1.b)