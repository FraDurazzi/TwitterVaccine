from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV,RidgeCV,RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re,os
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,matthews_corrcoef
from scipy.stats import bootstrap
from dirs import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
from Classification import WORKING_PATH


def plotting(x,y,text_comp,x_errs=None,y_errs=None,name=None,x_axis=None,y_axis=None):
    text_val,text_err=text_comp
    fig, ax = plt.subplots()

    # Customize error bars plot
    ax.errorbar(x, y, yerr=y_errs, fmt='o', ecolor='red', capsize=5, capthick=2, elinewidth=2, 
            marker='s', markersize=8, markerfacecolor='blue', markeredgewidth=2, markeredgecolor='black')

    ax.errorbar(["text"], text_val, yerr=np.array(text_err).T, fmt='o', ecolor='blue', capsize=5, capthick=2, elinewidth=2, 
            marker='s', markersize=8, markerfacecolor='yellow', markeredgewidth=2, markeredgecolor='black')

    # Add titles and labels
    ax.set_title(name, fontsize=16, fontweight='bold')
    ax.set_ylabel(y_axis, fontsize=14)

    # Customize grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a legend
    #ax.legend(['Data with Error Bars'], loc='upper left', fontsize=12)

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=9.5)

    # Save the plot as a PDF
    plt.savefig(WORKING_PATH+name+".pdf", format='pdf')

    # Show the plot
    plt.show()

def reading(dictionary,feature,select="single"):
    if select=="single":
        value=np.empty(shape=9)
        ers=np.empty(shape=(9,2))
        for i in range(len(value)):
            value[i]=dictionary[list(dictionary.keys())[i]]["Val set"][feature]
            ers[i]=np.abs(list(dictionary[list(dictionary.keys())[i]]["Val set"]["int_conf_"+feature].values())-value[i])
    elif select=="comb":
        value=np.empty(shape=9)
        ers=np.empty(shape=(9,2))
        for i in range(len(value)):
            value[i]=dictionary[list(dictionary.keys())[10+i]]["Val set"][feature]
            ers[i]=np.abs(list(dictionary[list(dictionary.keys())[10+i]]["Val set"]["int_conf_"+feature].values())-value[i])
    elif select=="text":
        value=np.empty(shape=1)
        ers=np.empty(shape=(1,2))
        value[0]=dictionary[list(dictionary.keys())[9]]["Val set"][feature]
        ers[0]=[np.abs(i-value[0]) for i in list(dictionary[list(dictionary.keys())[9]]["Val set"]['int_conf_'+feature].values())]
    return value,ers

def main():
    f=open(DATA_DIR+"output_l2_lbfgs_basic_2l.json")
    dictionary=json.load(f)
    single_acc,single_acc_ers=reading(dictionary,"accuracy",select="single")
    single_f1,single_f1_ers=reading(dictionary,"f1_score",select="single")
    single_matt,single_matt_ers=reading(dictionary,"matthews",select="single")
    comb_acc,comb_acc_ers=reading(dictionary,"accuracy",select="comb")
    comb_f1,comb_f1_ers=reading(dictionary,"f1_score",select="comb")
    comb_matt,comb_matt_ers=reading(dictionary,"matthews",select="comb")
    text_acc,text_acc_ers=reading(dictionary,"accuracy",select="text")
    text_f1,text_f1_ers=reading(dictionary,"f1_score",select="text")
    text_matt,text_matt_ers=reading(dictionary,"matthews",select="text")
    single=[s.replace(" + ","\n").replace("_", "\n") for s in list(dictionary)[0:9]]
    comb=[s.replace(" + ","\n").replace("_", "\n") for s in list(dictionary)[10:]]
    print(text_acc.shape)
    print(text_acc_ers.shape)
    plotting(comb,comb_acc,y_errs=comb_acc_ers.transpose(),text_comp=(text_acc,text_acc_ers),name='Feature+Text Accuracy',x_axis='Different Feature',y_axis='Accuracy')
    plotting(comb,comb_f1,y_errs=comb_f1_ers.transpose(),text_comp=(text_f1,text_f1_ers),name='Feature+Text F1 Score',x_axis='Different Feature',y_axis='F1 Score')
    plotting(comb,comb_matt,y_errs=comb_matt_ers.transpose(),text_comp=(text_matt,text_matt_ers),name='Feature+Text Matthews Coeficient',x_axis='Different Feature',y_axis='Matthews Coeficient')
    plotting(single,single_acc,y_errs=single_acc_ers.transpose(),text_comp=(text_acc,text_acc_ers),name='Single Feature  Accuracy',x_axis='Different Feature',y_axis='Accuracy')
    plotting(single,single_f1,y_errs=single_f1_ers.transpose(),text_comp=(text_f1,text_f1_ers),name='Single Feature  F1 Score',x_axis='Different Feature',y_axis='F1 Score')
    plotting(single,single_matt,y_errs=single_matt_ers.transpose(),text_comp=(text_matt,text_matt_ers),name='Single Feature  Matthews Coeficient',x_axis='Different Feature',y_axis='Matthews Coeficient')




if __name__ == "__main__":
    main()