"""
The program is designed for preprocessing a dataset and preparing it for machine learning tasks. 
It involves reading data from multiple sources, merging them into a single DataFrame, performing undersampling to balance label classes, and splitting the data into training, testing, and validation sets. The resulting datasets are then saved as CSV files.
Data Loading and Merging:
    Read the main dataset, communities dataset, and positions dataset.
    Merge these datasets into a single DataFrame.
Preprocessing:
    Perform text cleaning and preprocessing on relevant columns.
    Balance label classes using undersampling.
    Map labels to numerical values.
Train-Test-Validation Split:
    Split the preprocessed data into training, testing, and validation sets
"""

import pandas as pd
import numpy as np
import seaborn as sns
import json
import re,os
from typing import Union
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
import pathlib
from build_graphs import DEADLINES
from load_embeddings import load
labels=['ProVax','AntiVax','Neutral']
random_state=42

def undersampling(df: pd.DataFrame, random_state : Union[int,None] =None) -> pd.DataFrame:
    """
    Perform undersampling on a DataFrame to balance label classes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'label' column.
    - random_state (int|None): Seed for reproducibility in random sampling, if no seed is passed then completly randomic process is performed.

    Returns:
    - pd.DataFrame: Undersampled DataFrame with balanced label classes.

    """
    l = df.label.value_counts()['ProVax']
    out = pd.concat([
        df[df.label == 'ProVax'],
        df[df.label == 'AntiVax'].sample(l, random_state=random_state),
        df[df.label == 'Neutral'].sample(l, random_state=random_state)
    ], ignore_index=False)
    return out

def rescale(df: pd.DataFrame, columns: list[str] = ["fa2_x", "fa2_y"]) -> pd.DataFrame:
    """
    Rescale selected columns in the DataFrame using StandardScaler.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame containing the columns to be rescaled.
    columns : list of str, optional
        Names of the columns to be rescaled. Default is ["fa2_x", "fa2_y"].

    Returns:
    --------
    pd.DataFrame
        DataFrame with the specified columns rescaled using StandardScaler.

    Notes:
    ------
    This function uses StandardScaler from scikit-learn to standardize the features by removing the mean and scaling
    to unit variance.
    """
    # Initialize StandardScaler
    rescale = StandardScaler()
    
    # Fit the scaler to the specified columns
    rescale.fit(df[columns])
    
    # Transform and return the scaled columns
    return rescale.transform(df[columns])




def reading_merging(path_df: str,
                    name_df: list,
                    dtype_df: dict,
                    path_com: str,
                    names_com: list,
                    dtype_com: dict,
                    
                                    ) -> pd.DataFrame:
    """
    Read and merge data from multiple sources into a single DataFrame.

    Parameters:
    - path_df (str): Filepath for the main DataFrame CSV file.
    - dtype_df (dict): Data types for columns in the main DataFrame.
    - names_df (list): Column names for the main DataFrame.
    - path_com (str): Filepath for the community DataFrame CSV file.
    - names_com (list): Column names for the community DataFrame.
    - dtype_com (dict): Data types for columns in the community DataFrame.

    Returns:
    - pd.DataFrame: Merged DataFrame containing data from the main DataFrame, community DataFrame, and position data.
    """
    df = pd.read_csv(
        path_df,
        index_col="id",
        #names=name_df,
        dtype=dtype_df,
        na_values=["", "[]"],
        parse_dates=["created_at"],
        lineterminator="\n",
        )
    df_com = pd.read_csv(
        path_com,
        names=names_com,
        dtype=dtype_df,
        lineterminator="\n"
    )
    df_com=df_com[["user.id","leiden_90","louvain_90"]]
    df_com["user.id"]
    df_com_leiden= pd.read_csv(
    COMPATH/"communities_freq_leiden_90_2021-06-01.csv.gz",
    lineterminator="\n",
    names=["user.id","ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"],
    dtype={"user.id":str,"ld_0":int,"ld_1":int,"ld_2":int,"ld_3":int,"ld_4":int,"ld_5":int},
    header=0)
    df_com_leiden.set_index("user.id")
    df_com_louvain= pd.read_csv(
    COMPATH/"communities_freq_louvain_90_2021-06-01.csv.gz",
    lineterminator="\n",
    names=["user.id","lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6"],
    dtype={"user.id":str,"lv_0":int,"lv_1":int,"lv_2":int,"lv_3":int,"lv_4":int,"lv_5":int,"lv_6":int},
    header=0)
    df_com_louvain.set_index("user.id")
    df_futures=pd.read_csv(DATA_DIR+"tw2polarity_class_future.csv",
                           header=0,
                           index_col="id",
                           names=["id","annotation"],
                           dtype={"id":str,"annotation":str},
                           lineterminator="\n",
                           na_values=["","[]"])
    
    with open(DATA_DIR+'correct_future_pos.json') as f: 
        positions_fut = json.load(f)     
    # reconstructing the data as a dictionary 
    df_pos_fut=pd.DataFrame.from_dict(positions_fut, orient='index',columns=["fa2_x","fa2_y"])
    df_com_leiden_fut= pd.read_csv(
    COMPATH/"futures_communities_freq_leiden_90_2021-06-01.csv.gz",
    lineterminator="\n",
    names=["user.id","ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"],
    dtype={"user.id":str,"ld_0":int,"ld_1":int,"ld_2":int,"ld_3":int,"ld_4":int,"ld_5":int},
    header=0)
    df_com_louvain_fut= pd.read_csv(
    COMPATH/"futures_communities_freq_louvain_90_2021-06-01.csv.gz",
    lineterminator="\n",
    names=["user.id","lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6"],
    dtype={"user.id":str,"lv_0":int,"lv_1":int,"lv_2":int,"lv_3":int,"lv_4":int,"lv_5":int,"lv_6":int},
    header=0)
    df_future=df[df.index.isin(df_futures.index)].copy()
    df_future["annotation"]=df_future.index.map(df_futures["annotation"])
    df=df[["text","annotation","user.id"]]
    #new features from load_embeddings
    n2v=load("n2v",deadline="2021-06-01")
    n2v=n2v.rename(columns={i:"n2v_"+str(i) for i in n2v.columns})
    n2v.index=n2v.index.map(str)
    lap=load("laplacian",deadline="2021-06-01")
    lap=lap.rename(columns={i:"lap_"+str(i) for i in lap.columns})
    lap.index=lap.index.map(str)
    fa2=load(kind='fa2',deadline="2021-06-01")
    fa2.index=fa2.index.map(str)
    #df_anno_future.to_csv(DATA_DIR+"full_futures_annotated.csv",lineterminator='\n')
    df_future=df_future.merge(df_com_louvain_fut.set_index("user.id"),how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(df_com_leiden_fut.set_index("user.id"),how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(df_pos_fut,how="left",left_on="user.id",right_index=True)
    df=df.merge(df_com_leiden.set_index("user.id"),how="left",left_on="user.id",right_index=True)
    df=df.merge(df_com_louvain.set_index("user.id"),how="left",left_on="user.id",right_index=True)
    df=df.merge(df_com.set_index("user.id"),how="left",left_on="user.id",right_index=True)
    df=df.merge(n2v,how="left",left_on="user.id",right_index=True)
    df=df.merge(fa2,how="left",left_on="user.id",right_index=True)
    df=df.merge(lap,how="left",left_on="user.id",right_index=True)

    return df,df_future

def embedding(df: pd.DataFrame,
              model_name: str = 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0') -> pd.DataFrame:
    """
    Compute sentence embeddings using a pre-trained transformer model.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame containing the sentences for which embeddings are to be computed.
    model_name : str, optional
        Name or path of the pre-trained transformer model to be used for computing embeddings.
        Default is 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional columns containing the computed embeddings for each sentence.

    Notes:
    ------
    This function utilizes the SentenceTransformer library to encode sentences into fixed-dimensional vectors
    using pre-trained transformer models.
    """
    # Load the pre-trained transformer model
    model = SentenceTransformer(model_name)
    
    # Define column names for embedding columns
    list_cols = ["emb_col_" + str(i) for i in range(768)]
    
    # Copy the input DataFrame to avoid modifying the original
    df_out = df.copy()
    
    # Compute embeddings for each sentence in the DataFrame
    df_out[list_cols] = model.encode(df_out["sentence"], show_progress_bar=True)
    
    return df_out

def preproc(df: pd.DataFrame,
            df_fut:pd.DataFrame,
            label: list,
            seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocess DataFrame for text classification.
    Missing values in the text will be replaced, unused columns will be removed, and used columns will be renamed.
    Type checking for the columns (e.g leiden_90 and louvain_90 are fixed to int).
    Label column are mapped to int.
    Undersampling is applied.
    

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing text data and annotations.
    - label (list): List of labels for classification.
    - seed (int): Random seed for undersampling.

    Returns:
    - tuple[pd.DataFrame, np.ndarray]: tuple containing the preprocessed DataFrame and corresponding indices.
    """
    df_anno=df[df['annotation'].notna()].copy()
    df_anno.loc[:,'text']=df_anno['text'].apply(lambda x: x.replace('\n',' ') #Unix newline character
                                                            .replace('\t','') #Tab character
                                                            .replace("\r\n"," ") #Windows newline character
                                                            .replace('\u0085'," ") #Unicode nextline character
                                                            .replace('\u2028'," ") #Unicone line separator character
                                                            .replace('\u2029'," ")) #Unicode paragraph separator character
    df_anno=df_anno.rename(columns={'text':'sentence','annotation':'label'})
    for i in ["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6","ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"]:
        df_anno[i]=df_anno[i].apply(int)
    leid_sum=df_anno[["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"]].sum(axis=1)
    louv_sum=df_anno[["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6"]].sum(axis=1)
    for i in ["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6"]:
        df_anno[i]=df_anno[i].divide(louv_sum)
    for i in ["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"]:
        df_anno[i]=df_anno[i].divide(leid_sum)
    label2id = {label[0]:0, label[1]:1, label[2]:2}
    df_anno=undersampling(df_anno,seed)
    ids=df_anno.index.to_numpy()
    if(len(label)==2):
        label2id = {label[0]:0, label[1]:np.nan, label[2]:1}
    df_anno["label"]=df_anno["label"].map(label2id).dropna()
    df_anno["label"]=df_anno["label"].apply(int)
    df_anno[["fa2_x", "fa2_y"]]=rescale(df_anno,["fa2_x", "fa2_y"])
    #PREPROCESSING ON SECOND DATASET
    df_fut.loc[:,'text']=df_fut['text'].apply(lambda x: x.replace('\n',' ') #Unix newline character
                                                            .replace('\t','') #Tab character
                                                            .replace("\r\n"," ") #Windows newline character
                                                            .replace('\u0085'," ") #Unicode nextline character
                                                            .replace('\u2028'," ") #Unicone line separator character
                                                            .replace('\u2029'," ")) #Unicode paragraph separator character
    df_fut=df_fut[["text",
                   "annotation",
                   "lv_0",
                   "lv_1",
                   "lv_2",
                   "lv_3",
                   "lv_4",
                   "lv_5",
                   "lv_6",
                   "ld_0",
                   "ld_1",
                   "ld_2",
                   "ld_3",
                   "ld_4",
                   "ld_5",
                   "fa2_x",
                   "fa2_y"]].rename(columns={'text':'sentence','annotation':'label'})
    for i in ["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6","ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"]:
        df_fut[i]=df_fut[i].apply(int)
    leid_sum=df_fut[["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"]].sum(axis=1)
    louv_sum=df_fut[["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6"]].sum(axis=1)
    for i in ["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6"]:
        df_fut[i]=df_fut[i].divide(louv_sum)
    for i in ["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5"]:
        df_fut[i]=df_fut[i].divide(leid_sum)
    if(len(label)==2):
        label2id = {label[0]:0, label[1]:np.nan, label[2]:1}
    df_fut["label"]=df_fut["label"].map(label2id).dropna()
    df_fut["label"]=df_fut["label"].apply(int)
    df_fut[["fa2_x'","fa2_y'"]]=rescale(df_fut)
    return (embedding(df_anno),ids,embedding(df_fut))

def main(DATA_INFO):
    path_df,name_df,dtype_df,path_com,names_com,dtype_com,seed,label,DATA_PATH=DATA_INFO
    df,df_fut=reading_merging(path_df,name_df,dtype_df,path_com,names_com,dtype_com)
    df,ids,df_fut=preproc(df,df_fut,labels,seed)
    id_train,id_test=train_test_split(ids, test_size=0.33, random_state=42)
    id_test,id_val=train_test_split(ids, test_size=0.5, random_state=42)
    df[df.index.isin(id_train)].to_csv(DATA_PATH+'train.csv',lineterminator='\n')
    df[df.index.isin(id_test)].to_csv(DATA_PATH+'test.csv',lineterminator='\n')
    df[df.index.isin(id_val)].to_csv(DATA_PATH+'val.csv',lineterminator='\n')
    df_fut.to_csv(DATA_PATH+'fut.csv',lineterminator='\n')

if __name__ == "__main__":
    path_df=LARGE_DATA_DIR+"df_full.csv.gz"
    name_df=['created_at', 
             'text', 
             'user.id',              
             'user.screen_name', 
             'place', 
             'url',       
             'retweeted_status.id', 
             'retweeted_status.user.id',       
             'retweeted_status.url', 
             'annotation', 
             'user_annotation', 
             'lang',       
             'leiden_90', 
             'louvain_90', 
             'fa2_x', 
             'fa2_y']
    dtype_df={
            "id": str,
            "text": str,
            "user.id": str,
            "user.screen_name": str,
            "place": str,
            "url": str,
            "retweeted_status.id": str,
            "retweeted_status.user.id": str,
            "retweeted_status.url": str,
            "annotation": str,
            "user_annotation": str,
            "lang": str,
        }
    path_com=DATA_DIR+"coms/communities_2021-06-01.csv.gz"
    dtype_com={"user.id":str,
           "leiden":int,
           "infomap":int,
           "louvain":int,
           "leiden_5000":int,
           "leiden_90":int,
           "louvain_5000":int,
           "louvain_90":int,
           "infomap_5000":int,
           "infomap_90":int}
    names_com=["user.id","leiden","infomap","louvain","leiden_5000","leiden_90","louvain_5000","louvain_90","infomap_5000","infomap_90"]
    COMPATH=pathlib.Path(DATA_DIR+"coms")
    COMPATH.mkdir(parents=True, exist_ok=True)
    DATA_INFO=(path_df,name_df,dtype_df,path_com,names_com,dtype_com,random_state,labels,DATA_DIR)
    main(DATA_INFO)




    