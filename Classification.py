from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression,RidgeCV,RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import re,os
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,matthews_corrcoef
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
#label_list=[0,1,2]

def compute_metrics(predictions,labels):
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'weighted')
    f1s= f1_score(labels, predictions, average = None)
    matt=matthews_corrcoef(labels, predictions)
    return {'accuracy': acc, 'f1_score': f1, 'f1_scores': f1s,'matthews':matt}

def loader(kind):
    """
    kind can be train,val,test,fut
    """
    df = pd.read_csv(DATA_DIR+kind+".csv",
                     #index_col="Unnamed: 0	",
                     na_values=["", "[]"],
                     lineterminator="\n")
    df.rename(columns={"Unnamed: 0":"id"},inplace=True)
    df=df.set_index("id")
    return df


def main():
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
    rescale=StandardScaler()
    rescale.fit(train_df[using_cols])
    train_df[using_cols]=rescale.transform(train_df[using_cols])
    val_df[using_cols]=rescale.transform(val_df[using_cols])
    test_df[using_cols]=rescale.transform(test_df[using_cols])
    fut_df[using_cols]=rescale.transform(fut_df[using_cols])
    ###
    clf=LogisticRegression(penalty=penalty,solver=solver,random_state=42,max_iter=10000).fit(train_df[using_cols],train_df["label"])
    try:
        f=open(DATA_DIR+filename,"x") 
        f.write(using+":\n")
    except FileExistsError:
        f=open(DATA_DIR+filename,"a")
        f.write(using+":\n")
    f.write("\t train test: "+ str(compute_metrics(clf.predict(train_df[using_cols]),train_df["label"]))+"\n")
    f.write("\t val test: "+ str(compute_metrics(clf.predict(val_df[using_cols]),val_df["label"]))+"\n")
    f.write("\t test test: "+ str(compute_metrics(clf.predict(test_df[using_cols]),test_df["label"]))+"\n")
    f.write("\t fut test: "+ str(compute_metrics(clf.predict(fut_df[using_cols]),fut_df["label"]))+"\n")
    f.close()


if __name__ == "__main__":
    features=["lv_0","lv_1","lv_2","lv_3","lv_4","lv_5","lv_6","ld_0","ld_1","ld_2","ld_3","ld_4","ld_5","x_pos","y_pos"]
    n2v=['n2v_1', 'n2v_2', 'n2v_3', 'n2v_4','n2v_5', 'n2v_6', 'n2v_7', 'n2v_8']
    leiden=['ld_0', 'ld_1', 'ld_2', 'ld_3', 'ld_4', 'ld_5','ld_6', 'ld_7']
    louvain=['lv_1', 'lv_2', 'lv_3', 'lv_4', 'lv_5', 'lv_6','lv_7']
    lap=['lap_1', 'lap_2','lap_3', 'lap_4', 'lap_5', 'lap_6', 'lap_7', 'lap_8', 'lap_9', 'lap_10']
    fa2=['fa2_x', 'fa2_y']
    lab_prop=['lab_prop_0', 'lab_prop_1', 'lab_prop_2', 'lab_prop_3','lab_prop_4', 'lab_prop_5', 'lab_prop_6', 'lab_prop_7', 'lab_prop_8','lab_prop_9']
    norm_lap=['norm_lap1', 'norm_lap2', 'norm_lap3', 'norm_lap4','norm_lap5', 'norm_lap6', 'norm_lap7', 'norm_lap8', 'norm_lap9','norm_lap10']
    norm_leiden=['norm_ld_0', 'norm_ld_1', 'norm_ld_2', 'norm_ld_3', 'norm_ld_4', 'norm_ld_5','norm_ld_6', 'norm_ld_7']
    norm_louvain=['norm_lv_1', 'norm_lv_2', 'norm_lv_3', 'norm_lv_4', 'norm_lv_5', 'norm_lv_6','norm_lv_7']
    list_cols=["emb_col_"+str(i) for i in range(768)]
    penalty="l1"
    solver='saga'
    using="fa2"
    using_cols=fa2
    labels=[0,1,2]
    main()