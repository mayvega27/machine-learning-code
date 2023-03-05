import numpy as np
import pandas as pd

# data_dict = {-1:np.array([[1,7],
#                           [2,8],
#                           [3,8],]),
             
#              1:np.array([[5,1],
#                          [6,-1],
#                          [7,3],])}

# all_data = []
# for yi in data_dict:
#     for featureset in data_dict[yi]:
#         for feature in featureset:
#             all_data.append(feature)

# print(type(all_data), all_data)

# all_data = list(np.array(data_dict[-1]).flatten())+list(np.array(data_dict[1]).flatten())
# print(type(all_data), all_data)

# df = pd.DataFrame.from_dict(data_dict)
# print(df)


# mydictionary = {'physics': [68, 74, 77, 77],
#      'chemistry': [84, 56, 73, 69],
#      'algebra': [78, 88, 82, 87]}

# # create dataframe
# df_marks = pd.DataFrame(mydictionary)

# max = np.max(df_marks.max())
# print(max, type(max))

# print(df_marks['physics'].unique(), type(df_marks['physics'].unique()))

# for i in df_marks['physics'].unique():
#     print(i, type(i))


# a = np.array([68, 74, 77, 77])
# print(np.unique(a), type(np.unique(a)))


data_dict = {"ft1": [1, 2, 3, 5, 6, 7], "ft2": [7, 8, 8, 1,-1,3], "y":[-1, -1, -1, 1, 1 ,1]}
    
print(len(data_dict))
# df = pd.DataFrame.from_dict(data_dict)
# X = df.copy().iloc[:, :-1].values
# y = df.copy().iloc[:, -1].values

# print(X.shape[1])

# # print("This is X", X, type(X), len(X))
# # print("This is y", y, type(y), y.shape)

# # print(np.amax(X), np.amin(X))

# #print(df)


# w = np.array([3,3])
# w_2 = np.full((X.shape[1],), 10)
# print(w.shape, w_2.shape)

from itertools import permutations

# def unique_permutations(iterable, r=None):
#     previous = tuple()
#     transforms = []
#     for p in permutations(sorted(iterable), r):
#        if p > previous:
#           previous = p
#           transforms.append(list(p))

#     return transforms  

# transforms = unique_permutations(6*[1] + 6*[-1], 6)
# print(transforms, len(transforms))
