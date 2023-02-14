#Merging all the result files in to one file and saving it as .csv 
import glob
import pandas as pd
import matplotlib.pyplot as plt


path = glob.glob('**.csv')
density =[]
radius = []
#taking the first file as the starting of the dataFarame
main_df = pd.read_csv(path[0])
for p in path[1:]:
    print (p)
    df2 = pd.read_csv(p)
    main_df = pd.concat([main_df, df2])

main_df.to_csv('combined.csv',index=False)