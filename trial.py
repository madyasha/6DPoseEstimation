import pandas as pd

lst1= [1,2,3]
lst2= [4,5,6]
lst3 = [7,8,9]
df= pd.DataFrame(list(map(list, zip(lst1,lst2,lst3))))
df.to_csv('random.csv') 