import pandas as pd 
import os


data_types = ['50k_class_10', '400k_class_10']
data_types = [ '400k_class_10']
data_types = [ '50k_kd_class_10']


def get_data_type(data_type):
    
    dir1 = 'result_retroformer_2026_03_03/{}/'.format(data_type,data_type)
    dir2 = 'result_retroformer_2026_03_03_processed/{}/'.format(data_type,data_type)
    os.makedirs(dir2,exist_ok=True)
    files = [f for f in os.listdir(dir1) if f.endswith('.hdf5')]
    files.sort()
    for file in files:
        # print(file)
        df = pd.read_hdf(dir1+file,key='table')
        path = dir2+file.split('.')[0]+'/'
        os.makedirs(path,exist_ok=True)

        for i,col in enumerate(df.columns):
            # print(col)
            # print(df[col])
            df[col].to_csv(path+'{:02d}_{}.csv'.format(i+1,col),index=None)
            # print('*'*20)


  
for data_type in data_types:
    print(data_type)
    get_data_type(data_type)
    print('Done with {}'.format(data_type))
