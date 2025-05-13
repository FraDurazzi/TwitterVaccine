import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Preprocessing import loading
from sklearn.preprocessing import StandardScaler
import pathlib
PATH="/home/PERSONALE/niccolo.barbieri3/histos"
pathlib.Path(PATH).mkdir()

def plot(df1,df2,name):
    printing=PATH+f"/{name}/"
    pathlib.Path(printing).mkdir()
    # Creare istogrammi per colonne corrispondenti
    for col in df1.columns:
        # Creare l'istanza di StandardScaler per df1
        rescale1 = StandardScaler()

        # Riformattare i dati in un array 2D
        df1[col] = rescale1.fit_transform(df1[col].values.reshape(-1, 1))  # Assicurati di usare .values se è un DataFrame
        rescale2 = StandardScaler()
        df2[col] = rescale2.fit_transform(df2[col].values.reshape(-1, 1))
        # Calcolo del primo e terzo quartile
        Q1 = df1[col].quantile(0.25)
        Q3 = df1[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calcola i limiti superiori e inferiori per considerare un valore un outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Rimuovi i dati che si trovano al di fuori di questi limiti
        df1 = df1[(df1[col] >= lower_bound) & (df1[col] <= upper_bound)]
        Q1 = df2[col].quantile(0.25)
        Q3 = df2[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calcola i limiti superiori e inferiori per considerare un valore un outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Rimuovi i dati che si trovano al di fuori di questi limiti
        df2 = df2[(df2[col] >= lower_bound) & (df2 [col] <= upper_bound)]
        plt.figure(figsize=(8, 6))
    
        # Istogramma per la colonna corrispondente di df1
        sns.histplot(df1[col],bins=50, color="blue", label=f'{col} pre', alpha=0.5, kde=False)
    
        # Istogramma per la colonna corrispondente di df2
        sns.histplot(df2[col],bins=50, color="red", label=f'{col} post', alpha=0.5, kde=False)
    
        plt.title(f'Overlapping of '+name+f': {col}')
        plt.xlabel(name)
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig(printing+f"{col}.pdf", format='pdf')
        plt.savefig(printing+f"{col}.png", format='png')
        #plt.show()

def main():
    names=["n2v","lap", "fa2","ld","lv","lab_prop","norm_lap"]
    pre={names[i]:val for i,val in enumerate(loading("pre"))}
    post={names[i]:val for i,val in enumerate(loading("post"))}
    for i in names:
        plot(pre[i],post[i],i)



if __name__ == "__main__":
    main()