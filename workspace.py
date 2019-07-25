import numpy as np
import pandas as pd

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = pd.DataFrame(a)

b.to_csv('b.csv', index=False)
