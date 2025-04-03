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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold 
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import re,os
import json 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,matthews_corrcoef
from scipy.stats import bootstrap
from pathlib import Path
from dirs import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
fold=True
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
penalty="l2"
solver='saga'
#solver="lbfgs"
method="basic"
#l1_ratios=0.75
l1_ratios=0
labels=[0,1,2]
#labels=[0,1]
model="logistic"
#model="ridge"
if model=="logistic":
    settings= penalty+"_"+solver+"_"+method +f"_{l1_ratios}" if penalty=="elasticnet" else penalty+"_"+solver+"_"+method
else:
    settings=model
if fold:
    settings=settings+"/fold"
else:
    settings=settings+"/bootstrap"
result_path=DATA_DIR+"results/"+settings+"/"+str(len(labels))+"l/"
logits_path=f"{DATA_DIR}logits/{penalty}_{solver}_{method}_{l1_ratios}/" if penalty=="elasticnet" else f"{DATA_DIR}logits/{penalty}_{solver}_{method}/"
Path(logits_path).mkdir(parents=True, exist_ok=True)
Path(result_path).mkdir(parents=True, exist_ok=True)
print("Choosen model: "+model)

def logit(prob, eps=1e-6):
    prob=np.array(prob)
    prob = np.clip(prob, eps, 1 - eps)  
    return np.log(prob / (1 - prob))

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
    f1 = f1_score(labels, predictions, average = 'macro')
    f1s= f1_score(labels, predictions, average = None)
    matt=matthews_corrcoef(labels, predictions)
    boot_int_acc=bootstrap((predictions,labels),
                           lambda x,y : np.mean(x==y),
                           method=method,
                           paired=True,
                           vectorized=False,random_state=42).confidence_interval
    boot_int_f1_score=bootstrap((predictions,labels),
                                lambda x,y :f1_score(x,y,average='macro'),
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
                                 vectorized=False,random_state=42).confidence_interval._asdict() for i in np.unique(labels)]
    return {'accuracy': acc,
            'int_conf_accuracy': boot_int_acc._asdict(),
            'f1_score': f1,
            'int_conf_f1_score': boot_int_f1_score._asdict(),
            'f1_scores': f1s.tolist(),
            'single_class_int_conf_f1_score': f1s_conf,
            'matthews':matt,
            'int_conf_matthews': boot_int_matt._asdict()}
    
def compute_metrics_k_fold(predictions: np.ndarray, labels: np.ndarray,method="bca") -> dict:
    """
    Computes various performance metric.

    Parameters:
        predictions (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels.

    Returns:
        dict: A dictionary containing accuracy, F1 scores, Matthews correlation coefficient.
    """
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'macro')
    f1s= f1_score(labels, predictions, average = None)
    matt=matthews_corrcoef(labels, predictions)

    return {'accuracy': acc,
            'f1_score': f1,
            'f1_scores': f1s.tolist(),
            'matthews':matt}
    
def bootstrap_class(train_df: pd.DataFrame, 
                    val_df: pd.DataFrame, 
                    test_df: pd.DataFrame, 
                    fut_df: pd.DataFrame,
                    using_cols: list, 
                    using: str,
                    model: str=model, 
                    l1_ratios: float = l1_ratios, 
                    solver: str = solver, 
                    random_state: int = 42, 
                    max_iter: int = 1000000,Saving=False
                   ) -> dict:
    """
    Trains a classification model using logistic regression or ridge classifier, 
    makes predictions on train, validation, test, and future datasets, 
    and computes metrics for each dataset.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset containing features and labels.
    
    val_df : pd.DataFrame
        Validation dataset containing features and labels.
    
    test_df : pd.DataFrame
        Test dataset containing features and labels.
    
    fut_df : pd.DataFrame
        Future dataset containing features and labels.
    
    model : str
        The type of model to use. Options are "logistic" for logistic regression or 
        any other string for ridge classifier.
    
    l1_ratios : float, optional (default=0)
        The Elastic-Net mixing parameter, with 0 <= l1_ratios <= 1. Only used if 
        model is "logistic".
    
    solver : str, optional (default='lbfgs')
        Algorithm to use in the optimization problem for logistic regression.
    
    random_state : int, optional (default=42)
        Seed used by the random number generator.
    
    max_iter : int, optional (default=10000)
        Maximum number of iterations taken for the solvers to converge.
    
    using_cols : list, optional (default=None)
        List of columns to use for training the model.
    
    using : str, optional (default='classification')
        Specifies the task type. Can be used to handle different tasks such as 
        'classification' or others.

    Returns:
    --------
    results : dict
        Dictionary containing the evaluation metrics for train, validation, test, 
        and future datasets.
    """
    rescale=StandardScaler()
    rescale.fit(train_df[using_cols])
    train_df[using_cols]=rescale.transform(train_df[using_cols])
    val_df[using_cols]=rescale.transform(val_df[using_cols])
    if model == "logistic":
        if l1_ratios:
            clf = LogisticRegressionCV(penalty=penalty,
                                       solver=solver,
                                       random_state=random_state,
                                       max_iter=max_iter,
                                       l1_ratios=[l1_ratios],
                                       cv=5).fit(train_df[using_cols], train_df["label"])
        else:
            clf = LogisticRegressionCV(penalty=penalty,
                                       solver=solver,
                                       random_state=random_state,
                                       max_iter=max_iter,
                                       cv=5).fit(train_df[using_cols], train_df["label"])
    elif model == "random_forest":
        clf = RandomForestClassifier(random_state=random_state).fit(train_df[using_cols], train_df["label"])
    elif model == "svm":
        clf = SVC(random_state=random_state).fit(train_df[using_cols], train_df["label"])
    elif model=="ridge":
            clf=RidgeClassifierCV(alphas=(0.01,0.1, 1.0, 10.0,100,500,1000)).fit(train_df[using_cols],train_df["label"])
    else:
            raise ValueError("Model not recognized. Please choose 'logistic', 'random_forest', or 'svm'.")
    train_df["prediction"]=clf.predict(train_df[using_cols])
    val_df["prediction"]=clf.predict(val_df[using_cols])
    test_df["prediction"]=clf.predict(test_df[using_cols])
    fut_df["prediction"]=clf.predict(fut_df[using_cols])
    results={"Train_set": compute_metrics(train_df["prediction"],train_df["label"],method),
             "Val_set":compute_metrics(val_df["prediction"],val_df["label"],method),
             "Test_set":compute_metrics(test_df["prediction"],test_df["label"],method),
             "Fut_set":compute_metrics(fut_df["prediction"],fut_df["label"],method)}
    if Saving==True:
        return results,clf
    else:
        return results

def kfold_class(fold_df: pd.DataFrame, 
                test_df: pd.DataFrame, 
                fut_df: pd.DataFrame,
                using_cols: list, 
                using: str,
                model: str=model, 
                l1_ratios: float = l1_ratios, 
                solver: str = solver, 
                random_state: int = 42, 
                max_iter: int = 10000000,
                n_split=5,
                Saving=False) -> dict:
    """
    Trains a classification model using logistic regression or ridge classifier with K-fold cross-validation,
    makes predictions on test and future datasets, and computes metrics for each dataset.

    Parameters:
    -----------
    fold_df : pd.DataFrame
        Dataset to be used for K-fold cross-validation containing features and labels.
    
    test_df : pd.DataFrame
        Test dataset containing features and labels.
    
    fut_df : pd.DataFrame
        Future dataset containing features and labels.
    
    model : str
        The type of model to use. Options are "logistic" for logistic regression or 
        any other string for ridge classifier.
    
    l1_ratios : float, optional (default=0)
        The Elastic-Net mixing parameter, with 0 <= l1_ratios <= 1. Only used if 
        model is "logistic".
    
    solver : str, optional (default='lbfgs')
        Algorithm to use in the optimization problem for logistic regression.
    
    random_state : int, optional (default=42)
        Seed used by the random number generator.
    
    max_iter : int, optional (default=10000)
        Maximum number of iterations taken for the solvers to converge.
    
    using_cols : list, optional (default=None)
        List of columns to use for training the model.
    
    using : str, optional (default='classification')
        Specifies the task type, such as 'classification'.

    Returns:
    --------
    results : Dict[str, Dict[str, Any]]
        Dictionary containing the evaluation metrics for train, validation, test, 
        and future datasets.
"""
    kf = KFold(n_split)
    x=fold_df.index.tolist()
    kf.get_n_splits(x)
    label=fold_df["label"].unique()
    partial_results_train={'accuracy':np.empty(shape=n_split),
                           'f1_score':np.empty(shape=n_split),
                           'f1_scores':np.empty(shape=(n_split,len(label))),
                           'matthews':np.empty(shape=n_split)}
    partial_results_val={'accuracy':np.empty(shape=n_split),
                           'f1_score':np.empty(shape=n_split),
                           'f1_scores':np.empty(shape=(n_split,len(label))),
                           'matthews':np.empty(shape=n_split)}
    for i, (train_index, val_index) in enumerate(kf.split(x)):
        train_df, val_df = fold_df.iloc[train_index].copy(), fold_df.iloc[val_index].copy()
        rescale=StandardScaler()
        rescale.fit(train_df[using_cols])
        train_df[using_cols]=rescale.transform(train_df[using_cols])
        val_df[using_cols]=rescale.transform(val_df[using_cols])
        if model == "logistic":
            if (l1_ratios):                
                clf=LogisticRegressionCV(penalty=penalty,
                                         solver=solver,
                                         random_state=random_state,
                                         max_iter=max_iter,
                                         l1_ratios=[l1_ratios],
                                         cv=5).fit(train_df[using_cols],train_df["label"])
            else:
                clf=LogisticRegressionCV(penalty=penalty,
                                         solver=solver,
                                         random_state=random_state,
                                         max_iter=max_iter,
                                         cv=5).fit(train_df[using_cols],train_df["label"])
        elif model == "random_forest":
            clf = RandomForestClassifier(random_state=random_state).fit(train_df[using_cols], train_df["label"])
        elif model == "svm":
            clf = SVC(random_state=random_state,probability=True).fit(train_df[using_cols], train_df["label"])
        elif model=="ridge":
            clf=RidgeClassifierCV(alphas=(0.01,0.1, 1.0, 10.0,100,500,1000)).fit(train_df[using_cols],train_df["label"])
        else:
            raise ValueError("Model not recognized. Please choose 'logistic', 'random_forest','ridge' or 'svm'.")
        train_pred=clf.predict(train_df[using_cols])
        val_pred=clf.predict(val_df[using_cols])
        prob=list(clf.predict_proba(val_df[using_cols]))
        logits=[logit(prob[i]) for i in range(len(prob))]
        pd.DataFrame(data={"id":val_df.index.to_list(),
                       "logits":logits}).to_csv(logits_path+f"folding/val_logits_fold_{i}_{using}.csv",index=False)
        esteem=compute_metrics_k_fold(train_pred,train_df["label"].to_numpy())
        for j in list(esteem.keys()):
            partial_results_train[j][i]=esteem[j]
        esteem=compute_metrics_k_fold(val_pred,val_df["label"].to_numpy())
        for j in list(esteem.keys()):
            partial_results_val[j][i]=esteem[j]
    result_train={}    
    result_val={} 
    for j in list(partial_results_train.keys()):
        result_train[j]=np.mean(partial_results_train[j],axis=0).tolist()
        std=np.std(partial_results_train[j],axis=0).tolist()
        try:
            result_train['int_conf_'+j]=[{"low":result_train[j][k]-std[k],"high":result_train[j][k]+std[k]}for k in range(len(std))]
        except(TypeError):
            result_train['int_conf_'+j]={"low":result_train[j]-std,"high":result_train[j]+std}        
        result_val[j]=np.mean(partial_results_val[j],axis=0).tolist()
        std=np.std(partial_results_val[j],axis=0).tolist()
        try:
            result_val['int_conf_'+j]=[{"low":result_val[j][k]-std[k],"high":result_val[j][k]+std[k]}for k in range(len(std))]
        except(TypeError):
            result_val['int_conf_'+j]={"low":result_val[j]-std,"high":result_val[j]+std}     
    if model == "logistic":
        if (l1_ratios):                
            clf=LogisticRegressionCV(penalty=penalty,
                                        solver=solver,
                                        random_state=random_state,
                                        max_iter=max_iter,
                                        l1_ratios=[l1_ratios],
                                        cv=5).fit(fold_df[using_cols],fold_df["label"])
        else:
            clf=LogisticRegressionCV(penalty=penalty,
                                        solver=solver,
                                        random_state=random_state,
                                        max_iter=max_iter,
                                        cv=5).fit(fold_df[using_cols],fold_df["label"])
    elif model == "random_forest":
        clf = RandomForestClassifier(random_state=random_state).fit(fold_df[using_cols], fold_df["label"])
    elif model == "svm":
        clf = SVC(random_state=random_state,probability=True).fit(fold_df[using_cols], fold_df["label"])
    elif model=="ridge":
        clf=RidgeClassifierCV(alphas=(0.01,0.1, 1.0, 10.0,100,500,1000)).fit(fold_df[using_cols],fold_df["label"])
    else:
        raise ValueError("Model not recognized. Please choose 'logistic', 'random_forest','ridge' or 'svm'.")
    test_df["prediction"]=clf.predict(test_df[using_cols])
    fut_df["prediction"]=clf.predict(fut_df[using_cols])
    prob=list(clf.predict_proba(fold_df[using_cols]))
    logits=[logit(prob[i]) for i in range(len(prob))]
    pd.DataFrame(data={"id":fold_df.index.to_list(),
                       "logits":logits}).to_csv(logits_path+f"train/train_logits_{using}.csv",index=False)
    prob=list(clf.predict_proba(test_df[using_cols]))
    logits=[logit(prob[i]) for i in range(len(prob))]
    pd.DataFrame(data={"id":test_df.index.to_list(),
                       "logits":logits}).to_csv(logits_path+f"test/test_logits_{using}.csv",index=False)
    prob=list(clf.predict_proba(fut_df[using_cols]))
    logits=[logit(prob[i]) for i in range(len(prob))]
    pd.DataFrame(data={"id":fut_df.index.to_list(),
                       "logits":logits}).to_csv(logits_path+f"fut/fut_logits_{using}.csv",index=False)
    results={"Train_set": result_train,
             "Val_set":result_val,
             "Test_set":compute_metrics(test_df["prediction"],test_df["label"],method),
             "Fut_set":compute_metrics(fut_df["prediction"],fut_df["label"],method)}
    if Saving==True:
        return results,clf
    else:
        return results

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
        fold_df=loader("fold_2l")
    else:
        train_df=loader("train")
        val_df=loader("val")
        test_df=loader("test")
        fut_df=loader("fut")
        fold_df=loader("fold")   
    try:
        f=open(result_path+filename,"x") 
    except FileExistsError:
        f=open(result_path+filename,"a")  
        ###Rescaling the used feature
        #if (len(using_cols)<11): se scommentata dare tab fino agli hastag sotto
        rescale=StandardScaler()
        rescale.fit(fold_df[using_cols])
        #train_df[using_cols]=rescale.transform(train_df[using_cols])
        #val_df[using_cols]=rescale.transform(val_df[using_cols])
        test_df[using_cols]=rescale.transform(test_df[using_cols])
        fut_df[using_cols]=rescale.transform(fut_df[using_cols])
        #fold_df[using_cols]=rescale.transform(fold_df[using_cols])
        ###
    if (not fold):
        results=bootstrap_class(train_df,val_df,test_df,fut_df,using=using,using_cols=using_cols)
    else:
        results=kfold_class(fold_df,test_df,fut_df,using=using,using_cols=using_cols)
    f.write(using+":\n")
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
    norm_lab_prop=['norm_lab_prop_0', 'norm_lab_prop_1', 'norm_lab_prop_2', 'norm_lab_prop_3']
    text_cols=["emb_col_"+str(i) for i in range(768)]
    #text_cols=["emb_col_"+str(i) for i in range(10)]
    features=[n2v,
              #leiden,
              #louvain,
              lap,
              fa2,
              #lab_prop,
              norm_lap,
              norm_leiden,
              norm_louvain,
              norm_lab_prop]
    features_name=["n2v",
                   #"leiden",
                   #"louvain",
                   "lap",
                   "fa2",
                   #"lab_prop",
                   "norm_lap",
                   "norm_leiden",
                   "norm_louvain",
                   "norm_lab_prop"]
    using="norm_lap"
    using_cols=norm_lap
    all_results={}
    filename="output.txt"
    fileout="output.json"
    f=open(result_path+fileout,"w")
    f.close()
    f=open(result_path+filename,"w")
    f.close()
    for i in range(len(features)):
        using=features_name[i]
        using_cols=features[i]
        all_results[using]=main()
    """
    using_cols=text_cols
    using="text"
    all_results[using]=main()
    for i in range(len(features)):
        using=features_name[i]+" + text"
        using_cols=np.append(features[i],text_cols)        
        all_results[using]=main()
    """
    try:
        f=open(result_path+fileout,"x") 
    except FileExistsError:
        f=open(result_path+fileout,"a")
    f=open(result_path+fileout,"w")
    json.dump(all_results,f)
    f.close()