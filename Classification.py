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
from sklearn.linear_model import LogisticRegressionCV,RidgeCV,RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import re,os
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,matthews_corrcoef
from scipy.stats import bootstrap
from dirs import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
#label_list=[0,1,2]

def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """
    Computes various performance metrics and their confidence intervals using bootstrapping.

    Parameters:
        predictions (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels.

    Returns:
        dict: A dictionary containing accuracy, F1 scores, Matthews correlation coefficient, and their confidence intervals.
    """
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'weighted')
    f1s= f1_score(labels, predictions, average = None)
    matt=matthews_corrcoef(labels, predictions)
    boot_int_acc=bootstrap((predictions,labels),
                           lambda x,y : np.mean(x==y),
                           method="percentile",
                           paired=True,
                           vectorized=False).confidence_interval
    boot_int_f1_score=bootstrap((predictions,labels),
                                lambda x,y :f1_score(x,y,average='weighted'),
                                method="percentile",
                                paired=True,
                                vectorized=False).confidence_interval
    boot_int_matt=bootstrap((predictions,labels),
                           matthews_corrcoef,
                           method="percentile",
                           paired=True,
                           vectorized=False).confidence_interval
    f1s_conf=[bootstrap((predictions[labels[labels==i].index],labels[labels[labels==i].index]),
                                 lambda x,y : f1_score(x,y,average="weighted"),
                                 method="percentile",
                                 paired=True,
                                 vectorized=False).confidence_interval._asdict() for i in labels.unique()]
    return {'accuracy': acc,
            'int_conf_accuracy': boot_int_acc._asdict(),
            'f1_score': f1,
            'int_conf_f1_score': boot_int_f1_score._asdict(),
            'f1_scores': f1s,
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
    if len(labels)==2:
        train_df=loader("train_2l")
        val_df=loader("val_2l")
        test_df=loader("test_2l")
        fut_df=loader("fut_2l")
        filename="output_2l.txt"
    else:
        train_df=loader("train")
        val_df=loader("val")
        test_df=loader("test")
        fut_df=loader("fut")
        filename="output.txt"
    ###Rescaling the used feature
    print(fut_df.columns)
    rescale=StandardScaler()
    rescale.fit(train_df[using_cols])
    train_df[using_cols]=rescale.transform(train_df[using_cols])
    val_df[using_cols]=rescale.transform(val_df[using_cols])
    test_df[using_cols]=rescale.transform(test_df[using_cols])
    fut_df[using_cols]=rescale.transform(fut_df[using_cols])
    ###
    clf=LogisticRegressionCV(penalty=penalty,solver=solver,random_state=42,max_iter=10000).fit(train_df[using_cols],train_df["label"])
    try:
        f=open(DATA_DIR+filename,"x") 
        f.write(using+":\n")
    except FileExistsError:
        f=open(DATA_DIR+filename,"a")
        f.write(using+":\n")
    predicted_train=clf.predict(train_df[using_cols])
    predicted_val=clf.predict(val_df[using_cols])
    predicted_test=clf.predict(test_df[using_cols])
    predicted_fut=clf.predict(fut_df[using_cols])
    train_df["prediction"]=predicted_train
    val_df["prediction"]=predicted_val
    test_df["prediction"]=predicted_test
    fut_df["prediction"]=predicted_fut
    f.write("\t train test: "+ str(compute_metrics(train_df["prediction"],train_df["label"]))+"\n")
    f.write("\t val test: "+ str(compute_metrics(val_df["prediction"],val_df["label"]))+"\n")
    f.write("\t test test: "+ str(compute_metrics(test_df["prediction"],test_df["label"]))+"\n")
    f.write("\t fut test: "+ str(compute_metrics(fut_df["prediction"],fut_df["label"]))+"\n")
    f.write("\n \n \n")
    f.close()
    


if __name__ == "__main__":
    n2v=['n2v_1', 'n2v_2', 'n2v_3', 'n2v_4','n2v_5', 'n2v_6', 'n2v_7', 'n2v_8']
    leiden=['ld_0', 'ld_1', 'ld_2', 'ld_3', 'ld_4', 'ld_5','ld_6', 'ld_7']
    louvain=['lv_1', 'lv_2', 'lv_3', 'lv_4', 'lv_5', 'lv_6','lv_7']
    lap=['lap_1', 'lap_2','lap_3', 'lap_4', 'lap_5', 'lap_6', 'lap_7', 'lap_8', 'lap_9', 'lap_10']
    fa2=['fa2_x', 'fa2_y']
    lab_prop=['lab_prop_0', 'lab_prop_1', 'lab_prop_2', 'lab_prop_3','lab_prop_4', 'lab_prop_5', 'lab_prop_6', 'lab_prop_7', 'lab_prop_8','lab_prop_9']
    norm_lap=['norm_lap_1', 'norm_lap_2', 'norm_lap_3', 'norm_lap_4','norm_lap_5', 'norm_lap_6', 'norm_lap_7', 'norm_lap_8', 'norm_lap_9','norm_lap_10']
    norm_leiden=['norm_ld_0', 'norm_ld_1', 'norm_ld_2', 'norm_ld_3', 'norm_ld_4', 'norm_ld_5','norm_ld_6', 'norm_ld_7']
    norm_louvain=['norm_lv_1', 'norm_lv_2', 'norm_lv_3', 'norm_lv_4', 'norm_lv_5', 'norm_lv_6','norm_lv_7']
    list_cols=["emb_col_"+str(i) for i in range(768)]
    features=[n2v,leiden,louvain,lap,fa2,lab_prop,norm_lap,norm_leiden,norm_louvain,list_cols]
    features_name=["n2v","leiden","louvain","lap","fa2","lab_prop","norm_lap","norm_leiden","norm_louvain","list_cols"]
    penalty="l1"
    solver='saga'
    using="norm_lap"
    using_cols=norm_lap
    labels=[0,1,2]
    """
    for i in range(len(features)):
        using=features_name[i]
        using_cols=features[i]
        print("Classification with:"+using)
        main()
    """
    main()