### Code by Adonis Silva Flores
### Special thanks to the developer of scikit-learn making machine learning easy!!!

print("Loading...")
# importing the required libraries/modules
from tkinter import *
from tkinter import filedialog, messagebox
import os
import time
import math
#import numpy as np
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#from IPython.display import display
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
#                              AdaBoostClassifier, GradientBoostingClassifier)
#from sklearn import svm, linear_model
#from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
#from collections import Counter
#from sklearn.preprocessing import StandardScaler, RobustScaler
#from sklearn.decomposition import PCA
#from sklearn.model_selection import cross_validate, KFold

# load sample data sets (in this case, a csv file) from scikit-learn
#from sklearn import datasets
#bcancer = datasets.load_breast_cancer()
print("Module ready!!!")




class Window(Frame):
  
    
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

            
    
    #Creation of init_window/s
    def init_window(self):            

        def Ibutton_pressed():
             
            # message box display
            messagebox.showinfo("Hopefully this helps!",
                                'This python program can be used to evaluate the performance'
                                ' of the different supervised machine learning classifiers.'
                                '\n \nListed in the SELECT DATA box are the publicly available'
                                ' datasets from scikit learn or you may select browse data to'
                                ' choose any data from file (csv files, comma delimited is the'
                                ' default where n_samples=n_rows, n_features=n_columns, and last'
                                ' column is the target class).'
                                '\n \nIn the SELECT CLASSIFIER box, you can choose the machine'
                                ' learning model to be tested.'
                                '\n \nAnd of course do not forget to press TRAIN button.'
                                '\n \nNote: depending on the size of your data, some models'
                                ' may take a ...while... to train.'
                                ' Just keep an eye on the progress ...')

        ##########################################################################################
                                   
        # where the training and selecting of data happens                            
        def Tbutton_pressed():

            # plotting module
            def PR_curve():
                precision_clf, recall_clf, thresholds_ = precision_recall_curve(y_test, y_pred)
                AvePre = average_precision_score(y_test, y_pred)
                fig1=plt.figure(figsize=(6, 5))
                plt.plot(recall_clf, precision_clf, 'C2', label='AvePrecision={0:0.2f}'.format(AvePre))
                plt.step(recall_clf, precision_clf, color='g', alpha=0.2, where='mid')
                plt.fill_between(recall_clf, precision_clf, interpolate=True, step='mid', color='g', alpha=0.2)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.xscale('linear')
                plt.title(ptitle)
                plt.legend()
                fig1.show()

            def Scatter_plot():
                # Plot the classifed data using the first 3 features ...
                fig2 = plt.figure(figsize=(6, 5))
                ax = Axes3D(fig2, rect=[0, 0.05, 0.9, 0.9], elev=48, azim=135)

                # Assigned the x,y,z axes of plot to features X0,X1,X2 respectively
                # That is corresponding to the first 3 features or the first 3 columns of the spreadsheet file
                X0, X1 , X2 = X_train[:, 0], X_train[:, 1], X_train[:, 2]
                ax.scatter(X0, X1, X2, c=y_train, alpha=1, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
                ax.set_xlabel('feature 1')
                ax.set_ylabel('feature 2')
                ax.set_zlabel('feature 3')
                ax.set_title('First three features of the data set')
                x_min, x_max = min(X0), max(X0)
                y_min, y_max = min(X1), max(X1)
                z_min, z_max = min(X2), max(X2)
                xx = np.arange(x_min, x_max)
                yy = np.arange(y_min, y_max)
                zz = np.arange(z_min, z_max)
                xxx, yyy, zzz = np.meshgrid(xx, yy, zz)
                fig2.show()

            def ROC_curve():
                fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                AUC = roc_auc_score(y_test, y_pred)
                fig3=plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, 'C2', label='AUC={0:0.2f}'.format(AUC))
                plt.step(fpr, tpr, color='g', alpha=0.2, where='mid')
                plt.fill_between(fpr, tpr, interpolate=True, step='mid', color='g', alpha=0.2)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.xscale('linear')
                plt.title(ptitle2)
                plt.legend()
                fig3.show()

                
            def plot_figs():

                if var1.get()==1 and var2.get()==0: 

                    # Plot PR curve ...
                    PR_curve()
                    # Plot the classifed data using the first 3 features ...
                    Scatter_plot()
                    print ('Output: Precision-Recall Curve and Scatter plot of selected three features')

                elif var1.get()==0 and var2.get()==1:

                    # Plot the PR curve ...
                    PR_curve()
                    # Plot ROC curve ...
                    ROC_curve()
                    print ('Output: Precision-Recall and ROC curves')

                elif var1.get()==1 and var2.get()==1:

                    # Plot the PR curve ...
                    PR_curve()
                    # Plot the classifed data using the first 3 features
                    Scatter_plot()
                    # Plot ROC curve ...
                    ROC_curve()
                    print ('Output: Precision-Recall, ROC curves and Scatter plot of selected three features')

                else:

                    # Plot the PR curve ...
                    PR_curve()
                    print ('Output: Precision-Recall Curve')

            #########################################################
                        
            if trainButton:
                         
                modelnum=choosemodel.curselection()
                datanum=dataselect.curselection()
                loadtext.delete('1.0', END)
                traintext.delete('1.0', END)
                print ("")

                                
                # selecting the data for training ...                    
                if datanum==(0,):
                    print ("Data loaded: breast cancer datasets")
                    loadtext.insert(INSERT, "breast cancer \ndatasets loaded")
                    X_data=bcancer['data']
                    y_target=bcancer['target']
                    
                elif datanum==(1,):
                                        
                    # Open file to load (csv files)
                    file_path = filedialog.askopenfilename(initialdir="C:/Users/Eigenaar/Desktop/python_prog",
                                                           filetypes=[('.csvfiles', '.csv')],
                                                           title='Select csv file for TRAINING')

                    input_data = file_path

                    # comma delimited is the default
                    df = pd.read_csv(input_data, header = 0)

                    # for space delimited use:
                    ## df = pd.read_csv(input_file, header = 0, delimiter = " ")

                    # for tab delimited use:
                    ## df = pd.read_csv(input_file, header = 0, delimiter = "\t")

                    # put the original column names in a python list
                    original_headers = list(df.columns.values)

                    # remove the non-numeric columns
                    df = df._get_numeric_data()

                    # put the numeric column names in a python list
                    numeric_headers = list(df.columns.values)

                    # create a numpy array with the numeric values for input into scikit-learn
                    idata = df.as_matrix()

                    # here last column of the csv file is the target data and the rest are the input data
                    X_loaded=idata[:, 0:(len(idata[0])-2)]
                    y_loaded=idata[:, len(idata[0])-1]

                    scaler = RobustScaler()
                    scaler.fit(X_loaded)
                    X_scaled = scaler.transform(X_loaded)

                    pca = PCA(n_components=int((len(idata[0])-2)*0.67))
                    pca.fit(X_scaled)
                    X_scaledPCA = pca.transform(X_scaled)

                    if var3.get()==1 and var4.get()==0:
                        print("Scaling applied")
                        X_data=X_scaled
                    
                    elif var3.get()==1 and var4.get()==1:
                        print("Scaling and PCA applied")
                        X_data=X_scaledPCA

                    elif var3.get()==0 and var4.get()==1:
                        print("Note: PCA not recommended without scaling!!!")
                        print("PCA not applied!!!")
                        tbox4.deselect()
                        X_data=X_loaded

                    else:
                    
                        X_data=X_loaded

                    # X_data=X_loaded
                    y_target=y_loaded

                    filename=os.path.split(file_path)[1]
                    print ("Data loaded: {}".format(filename))
                    loadtext.insert(INSERT, filename)
                    loadtext.insert(INSERT, "\nloaded")

                else:
                    print ("Error!!! NO data selected")
                    loadtext.insert(INSERT, "Error!!! \nNo data selected")

                # split data into training and test sets, default 75% training and 25% test set
                X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_target, test_size=0.25, random_state=42)

                traindataclasses_count = Counter(y_train.astype(int))
                print('Training data class', traindataclasses_count)
                
                testdataclasses_count = Counter(y_test.astype(int))
                print('Testing data class', testdataclasses_count)

                ###############################################################################################    
            
                traintext.delete('1.0', END)

                # selecting the model and plotting ...                      
                if modelnum==(0,):
                    print ("GradientBoosting is selected ... training in progress ...")
                    traintext.insert(INSERT, "training in progress!!!")
                                        
                    # use gradientboosting as classifier
                    clf = GradientBoostingClassifier(n_estimators=1000, verbose=2)

                    # fit the model to training data
                    clf.fit(X_train, y_train)
                                                                    
                    # evaluate the model using the test data
                    y_pred = clf.predict_proba(X_test)[:, 1]
                        
                    print ("training finished!!!")
                    traintext.delete('1.0', END)
                    traintext.insert(INSERT, "training with gradient \nboosting finished!!!")

                    # Plot title ...
                    ptitle='Precision-Recall Curve, GradientBoosting'
                    ptitle2='ROC Curve, GradientBoosting'
                                                                            
                elif modelnum==(1,):
                    print ("DecisionTree is selected ... training in progress ...")
                    traintext.insert(INSERT, "training in progress!!!")

                    # use decision tree as classifier
                    clf = DecisionTreeClassifier(max_depth=1000)

                    # fit the model to training data
                    clf.fit(X_train, y_train)
                                                                    
                    # evaluate the model using the test data
                    y_pred = clf.predict_proba(X_test)[:, 1]
                    
                    print ("training finished!!!")
                    traintext.delete('1.0', END)
                    traintext.insert(INSERT, "training with decisiontree finished!!!")

                    # Plot labels ...
                    ptitle='Precision-Recall Curve, DecisionTree'
                    ptitle2='ROC Curve, DecisionTree'
                                                            
                elif modelnum==(2,):
                    print ("RandomForest is selected ... training in progress ...")
                    traintext.insert(INSERT, "training in progress!!!")
                                        
                    # use random forest as classifier
                    clf = RandomForestClassifier(n_estimators=100, verbose=2)

                    # fit the model to training data
                    clf.fit(X_train, y_train)
                                                                    
                    # evaluate the model using the test data
                    y_pred = clf.predict_proba(X_test)[:, 1]
                        
                    print ("training finished!!!")
                    traintext.delete('1.0', END)
                    traintext.insert(INSERT, "training with random \nforest finished!!!")

                    # Plot labels ...
                    ptitle='Precision-Recall Curve, RandomForest'
                    ptitle2='ROC Curve, RandomForest'
                    
                elif modelnum==(3,):
                    print ("LogisticRegression is selected ... training in progress ...")
                    traintext.insert(INSERT, "training in progress!!!")

                    # use logistic regression as classifier
                    clf = linear_model.LogisticRegression(C=1, max_iter=200)

                    # fit the model to training data
                    clf.fit(X_train, y_train)
                                                                    
                    # evaluate the model using the test data
                    y_pred = clf.predict_proba(X_test)[:, 1]
                        
                    print ("training finished!!!")
                    traintext.delete('1.0', END)
                    traintext.insert(INSERT, "training with logisticregression finished!!!")

                    # Plot labels ...
                    ptitle='Precision-Recall Curve, LogisticRegression'
                    ptitle2='ROC Curve, LogisticRegression'
                                        
                elif modelnum==(4,):
                    print ("SVM with kernel=linear is selected ... training in progress ...")
                    traintext.insert(INSERT, "training in progress!!!")
                    
                    # use SVM as classifier
                    clf = svm.SVC(kernel='linear', C=1)

                    # fit the model to training data
                    clf.fit(X_train, y_train)
                                                                    
                    # evaluate the model using the test data
                    y_pred = clf.decision_function(X_test)
                        
                    print ("training finished!!!")
                    traintext.delete('1.0', END)
                    traintext.insert(INSERT, "training with SVM \nkernel:linear finished!!!")

                    # Plot labels ...
                    ptitle='Precision-Recall Curve, SVM(kernel:linear)'
                    ptitle2='ROC Curve, SVM(kernel:linear)'
                    
                elif modelnum==(5,):
                    print ("SVM with kernel=polynomial is selected ... training in progress ...")
                    traintext.insert(INSERT, "training in progress!!!")
                    
                    # use SVM poly as classifier
                    clf = svm.SVC(kernel='poly', degree=2, C=1)

                    # fit the model to training data
                    clf.fit(X_train, y_train)
                                                                    
                    # evaluate the model using the test data
                    y_pred = clf.decision_function(X_test)

                    print ("training finished!!!")
                    traintext.delete('1.0', END)
                    traintext.insert(INSERT, "training with SVM \nkernel:poly finished!!!")    
                    
                    # Plot labels ...
                    ptitle='Precision-Recall Curve, SVM(kernel:poly)'
                    ptitle2='ROC Curve, SVM(kernel:poly)'
                
                else:
                    print ("Error!!! No classifier selected")
                    traintext.insert(INSERT, "Error!!! \nNo classifier selected")

                # Plot the Precision-Recall curve and/or first 3 features of data
                plot_figs()
                
        #################################################################################################

        # elements for the gui window ....            
                    
        # changing the title of our master widget      
        self.master.title("Evaluating_Classifier_PR-curve")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # change background of the root window
        ## self.config(bg='lightgreen')

        # creating a button/listbox instance and placing on the window
        loadtext=Text (self, width=22, height=2, fg="green",
                       bg="black")
        loadtext.insert(INSERT, "")
        loadtext.place(x=390, y=27)

        choosemodel=Listbox (self, width=25, height=6, fg="white",
                             bg="black", selectmode="single",
                             exportselection=0)
        choosemodel.insert(0, "   GradientBoosting")
        choosemodel.insert(1, "   DecisionTree")
        choosemodel.insert(2, "   RandomForest")
        choosemodel.insert(3, "   LogisticRegression")
        choosemodel.insert(4, "   SVM(kernel:linear)")
        choosemodel.insert(5, "   SVM(kernel:poly)")
        choosemodel.place(x=133, y=49)

        choosemodeltext=Text (self, width=19, height=1, bd=0, fg="black",
                            bg="white")
        choosemodeltext.insert(INSERT, " SELECT CLASSIFIER")
        choosemodeltext.place(x=133, y=25)
        choosemodeltext.config(state=DISABLED)
        
        trainButton = Button(self, text="TRAIN", width=10, height=1,
                             fg="white", bg="blue", command=Tbutton_pressed)
        trainButton.place(x=303, y=32)
        
        traintext=Text (self, width=22, height=2, fg="green",
                        bg="black")
        traintext.insert(INSERT, "")
        traintext.place(x=390, y=68)

        infoButton = Button(self, text="READ ME", width=10, height=1, fg="white",
                            bg="green", command=Ibutton_pressed)
        infoButton.place(x=303, y=115)

        sponsortext=Text (self, width=22, height=2, fg="black",
                          bg="red", bd=0)
        sponsortext.insert(INSERT, " Python3.4:sk-learn \n code by:A.S.F")
        sponsortext.place(x=391, y=111)
        sponsortext.config(state=DISABLED)

        dataselect=Listbox (self, width=17, height=3, fg="white",
                                    bg="black", selectmode="single",
                                    exportselection=0)
        dataselect.insert(0, "   breastcancer")
        dataselect.insert(1, "   browse data")
        dataselect.place(x=21, y=49)

        dataselecttext=Text (self, width=13, height=1, bd=0, fg="black",
                             bg="white")
        dataselecttext.insert(INSERT, " SELECT DATA")
        dataselecttext.place(x=20, y=25)
        dataselecttext.config(state=DISABLED)

        var1 = IntVar()
        tbox1=Checkbutton(self, text="Scatter Plot?", variable=var1)
        tbox1.place(x=20, y=126)

        var2 = IntVar()
        tbox2=Checkbutton(self, text="ROC Curve?", variable=var2)
        tbox2.place(x=20, y=106)

        var3 = IntVar()
        tbox3=Checkbutton(self, text="Scaling?", variable=var3)
        tbox3.place(x=303, y=64)

        var4 = IntVar()
        tbox4=Checkbutton(self, text="PCA?", variable=var4)
        tbox4.place(x=303, y=84)

                 
                
root = Tk()

#size of the window
root.geometry("590x170")
root.resizable(0,0)

app = Window(root)
root.mainloop()


