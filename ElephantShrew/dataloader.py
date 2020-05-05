import pandas as pd
class Data_loader:
    def load(filename, start_row=0, end_row=10e6):
        df = pd.read_pickle(filename)
        df = df[start_row:min(len(df),end_row)]
        df = df.reset_index(drop=True)#.drop(['index'],axis=1)
        return df
