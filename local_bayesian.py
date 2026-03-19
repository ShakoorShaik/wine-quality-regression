from utils import preprocessing as pp
from utils import normalization as nz

# file name
red_file = 'winequality-red.csv'
white_file = 'winequality-white.csv'
output_file = 'winequality-modified.csv'

# preprocessing and train test split
X_train, y_train, X_test, y_test = pp(red_file, white_file)
normd_X_train, normed_X_test, _ = nz(X_train, X_test)

# 