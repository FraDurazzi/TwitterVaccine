"""
Overview:
    This Python script performs classification using Logistic Regression with Cross-Validation.
    It includes data loading, preprocessing, model training, prediction, and evaluation through metrics computation.
    The code handles different datasets and labels, and computes confidence intervals for the metrics using bootstrapping.

Modules and Packages:
    - sentence_transformers: For sentence embeddings.
    - sklearn.linear_model: Contains LogisticRegressionCV, RidgeCV, and RidgeClassifierCV for different types of regression and classification models.
    - sklearn.preprocessing: Contains StandardScaler for feature scaling.
    - numpy: For numerical operations.
    - pandas: For data manipulation.
    - re: For regular expressions.
    - os: For environment variable handling.
    - json: For JSON operations.
    - sklearn.metrics: For various evaluation metrics.
    - scipy.stats: For statistical computations, specifically bootstrapping.

Directory Constants:
    - TRANSFORMERS_CACHE_DIR: Directory for cached transformer models.
    - DATA_DIR: Directory for dataset files.
    - LARGE_DATA_DIR: Directory for large dataset files.
"""
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV,RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import re,os
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,matthews_corrcoef
from scipy.stats import bootstrap
from pathlib import Path
from dirs import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
penalty='elasticnet'
solver='saga'
method="basic"
l1_ratios=0.4
labels=[0,1,2]
model="logistic"
#model="ridge"
if model=="logistic":
    settings=penalty+"_"+solver+"_"+method
else:
    settings=model
WORKING_PATH=DATA_DIR+settings+"/"+str(len(labels))+"l/"
Path(WORKING_PATH).mkdir(parents=True, exist_ok=True)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray,method="bca") -> dict:
    """
    Computes various performance metrics and their confidence intervals using bootstrapping.

    Parameters:
        predictions (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels.

    Returns:
        dict: A dictionary containing accuracy, F1 scores, Matthews correlation coefficient, and their confidence intervals.
    """
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'micro')
    f1s= f1_score(labels, predictions, average = None)
    matt=matthews_corrcoef(labels, predictions)
    boot_int_acc=bootstrap((predictions,labels),
                           lambda x,y : np.mean(x==y),
                           method=method,
                           paired=True,
                           vectorized=False,random_state=42).confidence_interval
    boot_int_f1_score=bootstrap((predictions,labels),
                                lambda x,y :f1_score(x,y,average='micro'),
                                method=method,
                                paired=True,
                                vectorized=False,random_state=42).confidence_interval
    boot_int_matt=bootstrap((predictions,labels),
                           matthews_corrcoef,
                           method=method,
                           paired=True,
                           vectorized=False,random_state=42).confidence_interval
    f1s_conf=[bootstrap((predictions==i,labels==i),
                                 lambda x,y : f1_score(x,y),
                                 method=method,
                                 paired=True,
                                 vectorized=False,random_state=42).confidence_interval._asdict() for i in labels.unique()]
    return {'accuracy': acc,
            'int_conf_accuracy': boot_int_acc._asdict(),
            'f1_score': f1,
            'int_conf_f1_score': boot_int_f1_score._asdict(),
            'f1_scores': f1s.tolist(),
            'single_class_int_conf_f1_score': f1s_conf,
            'matthews':matt,
            'int_conf_matthews': boot_int_matt._asdict()}

def loader(kind: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file based on the provided kind.

    Parameters:
        kind (str): A string indicating the dataset type (for 3 label case: "train", "val", "test", "fut"; 
                                                          for 2 label case: "train_2l", "val_2l", "test_2l", "fut_2l").

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    df = pd.read_csv(DATA_DIR+kind+".csv",
                     #index_col="Unnamed: 0	",
                     na_values=["", "[]"],
                     lineterminator="\n")
    df.rename(columns={"Unnamed: 0":"id"},inplace=True)
    df=df.set_index("id")
    return df


def main():
    """
    Main function to perform classification, training, and evaluation.
    """
    print("Classification with:"+using)
    if len(labels)==2:
        train_df=loader("train_2l")
        val_df=loader("val_2l")
        test_df=loader("test_2l")
        fut_df=loader("fut_2l")
    else:
        train_df=loader("train")
        val_df=loader("val")
        test_df=loader("test")
        fut_df=loader("fut")
        
    filename="output.txt"
    fileout="output.json"
    ###Rescaling the used feature
    rescale=StandardScaler()
    rescale.fit(train_df[using_cols])
    train_df[using_cols]=rescale.transform(train_df[using_cols])
    val_df[using_cols]=rescale.transform(val_df[using_cols])
    test_df[using_cols]=rescale.transform(test_df[using_cols])
    fut_df[using_cols]=rescale.transform(fut_df[using_cols])
    ###
    if model=="logistic":
        if (l1_ratios):
            clf=LogisticRegressionCV(penalty=penalty,
                                     solver=solver,
                                     random_state=42,
                                     max_iter=10000,
                                     l1_ratios=[l1_ratios],
                                     cv=5).fit(train_df[using_cols],train_df["label"])
        else:
            clf=LogisticRegressionCV(penalty=penalty,
                                     solver=solver,
                                     random_state=42
                                     ,max_iter=10000,
                                     cv=5).fit(train_df[using_cols],train_df["label"])
    else:
        clf=RidgeClassifierCV().fit(train_df[using_cols],train_df["label"])
    try:
        f=open(WORKING_PATH+filename,"x") 
    except FileExistsError:
        f=open(WORKING_PATH+filename,"a")   
    f.write(using+":\n")
    train_df["prediction"]=clf.predict(train_df[using_cols])
    val_df["prediction"]=clf.predict(val_df[using_cols])
    test_df["prediction"]=clf.predict(test_df[using_cols])
    fut_df["prediction"]=clf.predict(fut_df[using_cols])
    results={"Train set": compute_metrics(train_df["prediction"],train_df["label"],method),
             "Val set":compute_metrics(val_df["prediction"],val_df["label"],method),
             "Test set":compute_metrics(test_df["prediction"],test_df["label"],method),
             "Fut set":compute_metrics(fut_df["prediction"],fut_df["label"],method)}
    for i in results.keys():
        f.write("\t "+ i+": \n")
        for j in results[i].keys():
            f.write("\t \t"+j+ " :" + str(results[i][j])+"\n \n")
    """
    f.write("\t train test: "+ str(compute_metrics(train_df["prediction"],train_df["label"]))+"\n")
    f.write("\t val test: "+ str(compute_metrics(val_df["prediction"],val_df["label"]))+"\n")
    f.write("\t test test: "+ str(compute_metrics(test_df["prediction"],test_df["label"]))+"\n")
    f.write("\t fut test: "+ str(compute_metrics(fut_df["prediction"],fut_df["label"]))+"\n")
    """
    f.write("\n \n \n")
    f.close()

    return(results)


if __name__ == "__main__":
    n2v=['n2v_1', 'n2v_2', 'n2v_3', 'n2v_4','n2v_5', 'n2v_6', 'n2v_7', 'n2v_8', 'n2v_9', 'n2v_10']
    leiden=['ld_0', 'ld_1', 'ld_2', 'ld_3', 'ld_4', 'ld_5','ld_6']
    louvain=['lv_1', 'lv_2', 'lv_3', 'lv_4', 'lv_5', 'lv_6','lv_7']
    lap=['lap_1', 'lap_2','lap_3', 'lap_4', 'lap_5', 'lap_6', 'lap_7', 'lap_8', 'lap_9', 'lap_10']
    fa2=['fa2_x', 'fa2_y']
    lab_prop=['lab_prop_0', 'lab_prop_1', 'lab_prop_2', 'lab_prop_3']
    norm_lap=['norm_lap_1', 'norm_lap_2', 'norm_lap_3', 'norm_lap_4','norm_lap_5', 'norm_lap_6', 'norm_lap_7', 'norm_lap_8', 'norm_lap_9','norm_lap_10']
    norm_leiden=['norm_ld_0', 'norm_ld_1', 'norm_ld_2', 'norm_ld_3', 'norm_ld_4', 'norm_ld_5','norm_ld_6']
    norm_louvain=['norm_lv_1', 'norm_lv_2', 'norm_lv_3', 'norm_lv_4', 'norm_lv_5', 'norm_lv_6','norm_lv_7']
    text_cols=["emb_col_"+str(i) for i in range(768)]
    features=[n2v,leiden,louvain,lap,fa2,lab_prop,norm_lap,norm_leiden,norm_louvain]
    features_name=["n2v","leiden","louvain","lap","fa2","lab_prop","norm_lap","norm_leiden","norm_louvain"]
    using="norm_lap"
    using_cols=norm_lap
    results={}
    for i in range(len(features)):
        using=features_name[i]
        using_cols=features[i]
        results[using]=main()
    using_cols=text_cols
    using="text"
    results[using]=main()
    for i in range(len(features)):
        using=features_name[i]+" + text"
        using_cols=np.append(features[i],text_cols)        
        results[using]=main()
    if len(labels):
        fileout="output_2l.json"
    else:
        fileout="output.json"
    try:
        f=open(WORKING_PATH+fileout,"x") 
    except FileExistsError:
        f=open(WORKING_PATH+fileout,"a")
    json.dump(results,f)
    f.close()