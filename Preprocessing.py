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


def loading(type: str) -> tuple:
    """
    Load and process various types of graph embeddings and clustering results.

    This function loads different types of graph-related data based on the specified deadline type. 
    The data includes Node2Vec embeddings, Laplacian embeddings, Force Atlas 2 embeddings, Leiden clustering, 
    Louvain clustering, Label Propagation clustering, and Normalized Laplacian embeddings. Each loaded dataset 
    is renamed to include a prefix indicating its type and the index is converted to string type.

    Parameters:
    -----------
    type : str
        The deadline type to be used for loading the data. This parameter is passed to the `load` function.

    Returns:
    --------
    tuple
        A tuple containing seven pandas DataFrames, each with their columns renamed and indices converted to strings:
        - n2v: Node2Vec embeddings
        - lap: Laplacian embeddings
        - fa2: Force Atlas 2 embeddings
        - ld: Leiden clustering results
        - lv: Louvain clustering results
        - lab_prop: Label Propagation clustering results
        - norm_lap: Normalized Laplacian embeddings
    """

    n2v = load("n2v", deadline=type)
    n2v = n2v.rename(columns={i: "n2v_" + str(i) for i in n2v.columns})
    n2v.index = n2v.index.map(str)

    lap = load("laplacian", deadline=type)
    lap = lap.rename(columns={i: "lap_" + str(i) for i in lap.columns})
    lap.index = lap.index.map(str)

    fa2 = load(kind='fa2', deadline=type)
    fa2.index = fa2.index.map(str)

    ld = load("leiden", deadline=type)
    ld = ld.rename(columns={i: "ld_" + str(i) for i in ld.columns})
    ld.index = ld.index.map(str)

    lv = load("louvain", deadline=type)
    lv = lv.rename(columns={i: "lv_" + str(i) for i in lv.columns})
    lv.index = lv.index.map(str)

    lab_prop = load("labelpropagation", deadline=type)
    lab_prop = lab_prop.rename(columns={i: "lab_prop_" + str(i) for i in lab_prop.columns})
    lab_prop.index = lab_prop.index.map(str)

    norm_lap = load("norm_laplacian", deadline=type)
    norm_lap = norm_lap.rename(columns={i: "norm_lap_" + str(i) for i in norm_lap.columns})
    norm_lap.index = norm_lap.index.map(str)

    return n2v, lap, fa2, ld, lv, lab_prop, norm_lap


def reading_merging(path_df: str,
                    name_df: list,
                    dtype_df: dict,                    
                                    ) -> pd.DataFrame:
    """
    Read and merge data from multiple sources into a single DataFrame.

    Parameters:
    - path_df (str): Filepath for the main DataFrame CSV file.
    - dtype_df (dict): Data types for columns in the main DataFrame.
    - names_df (list): Column names for the main DataFrame.

    Returns:
    - pd.DataFrame: Merged DataFrame containing data from the main DataFrame, community DataFrame, and position data.
    """
    df = pd.read_csv(path_df,
                     index_col="id",
                     #names=name_df,
                     dtype=dtype_df,
                     na_values=["", "[]"],
                     parse_dates=["created_at"],
                     lineterminator="\n",
        )
    df_futures=pd.read_csv(DATA_DIR+"tw2polarity_class_future.csv",
                           header=0,
                           index_col="id",
                           names=["id","annotation"],
                           dtype={"id":str,"annotation":str},
                           lineterminator="\n",
                           na_values=["","[]"])
    
    df_future=df[df.index.isin(df_futures.index)].copy()
    df_future["annotation"]=df_future.index.map(df_futures["annotation"])
    df[df["created_at"]<pd.Timestamp("2020-10-01" + "T00:00:00+02")].annotation=np.nan#removing annotation of old retweets
    df=df[["text","annotation","user.id"]]
    df_future=df_future[["text","annotation","user.id"]]
    #new features from load_embeddings
    old_n2v,old_lap,old_fa2,old_ld,old_lv,old_lab_prop,old_norm_lap=loading("pre")
    new_n2v,new_lap,new_fa2,new_ld,new_lv,new_lab_prop,new_norm_lap=loading("post")
    #df_com_leiden=df_com_leiden.set_index("user.id")
    #df_com_leiden.index=df_com_leiden.index.map(str)
    #df_anno_future.to_csv(DATA_DIR+"full_futures_annotated.csv",lineterminator='\n')
    df=df.merge(old_n2v,how="left",left_on="user.id",right_index=True)
    df=df.merge(old_lap,how="left",left_on="user.id",right_index=True)
    df=df.merge(old_fa2,how="left",left_on="user.id",right_index=True)
    df=df.merge(old_ld,how="left",left_on="user.id",right_index=True)
    df=df.merge(old_lv,how="left",left_on="user.id",right_index=True)
    df=df.merge(old_lab_prop,how="left",left_on="user.id",right_index=True)
    df=df.merge(old_norm_lap,how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(new_n2v,how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(new_lap,how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(new_fa2,how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(new_ld,how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(new_lv,how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(new_lab_prop,how="left",left_on="user.id",right_index=True)
    df_future=df_future.merge(new_norm_lap,how="left",left_on="user.id",right_index=True)
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
    leid_sum=df_anno[["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5","ld_6","ld_7"]].sum(axis=1)
    louv_sum=df_anno[["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6","lv_7"]].sum(axis=1)
    for i in ["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6","lv_7"]:
        df_anno["norm_"+i]=df_anno[i].divide(louv_sum)
    for i in ["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5","ld_6","ld_7"]:
        df_anno["norm_"+i]=df_anno[i].divide(leid_sum)
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
    df_fut=df_fut.rename(columns={'text':'sentence','annotation':'label'})
    leid_sum_fut=df_fut[["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5","ld_6","ld_7"]].sum(axis=1)
    louv_sum_fut=df_fut[["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6","lv_7"]].sum(axis=1)
    for i in ["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6","lv_7"]:
        df_fut["norm_"+i]=df_fut[i].divide(louv_sum_fut)
    for i in ["ld_0","ld_1","ld_2","ld_3","ld_4","ld_5","ld_6","ld_7"]:
        df_fut["norm_"+i]=df_fut[i].divide(leid_sum_fut)
    if(len(label)==2):
        label2id = {label[0]:0, label[1]:np.nan, label[2]:1}
    df_fut["label"]=df_fut["label"].map(label2id).dropna()
    df_fut["label"]=df_fut["label"].apply(int)
    df_fut[["fa2_x'","fa2_y'"]]=rescale(df_fut)
    #print("df")
    #print(df.columns)
    #print("df_fut")
    #print(df_fut.columns)
    return (embedding(df_anno),ids,embedding(df_fut))

def main(DATA_INFO):
    path_df,name_df,dtype_df,seed,label,DATA_PATH=DATA_INFO
    df,df_fut=reading_merging(path_df,name_df,dtype_df)
    df,ids,df_fut=preproc(df,df_fut,labels,seed)
    id_train,id_test=train_test_split(ids, test_size=0.33, random_state=42)
    id_test,id_val=train_test_split(ids, test_size=0.5, random_state=42)
    #print("df in main")
    #print(df.columns)
    #print("df_fut in main")
    #print(df_fut.columns)
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
    COMPATH=pathlib.Path(DATA_DIR+"coms")
    COMPATH.mkdir(parents=True, exist_ok=True)
    DATA_INFO=(path_df,name_df,dtype_df,random_state,labels,DATA_DIR)
    main(DATA_INFO)




    