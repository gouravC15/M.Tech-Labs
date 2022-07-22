import pandas as pd
import os

print("\n[****] Reading CSV")
PATH ='/Users/gouravchirkhare/PycharmProjects/Unet_segment/R-Forest/'
df = pd.read_csv(PATH+'RF_FILES/Features.csv')
#dependent variable
Y = df["Labels"].values
#independent variables
X = df.drop(labels = ["Labels"], axis=1) 

#Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
print("\n[+] DONE: Test Train Splitting")

print("\n[****] Training Started")
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 42)
model.fit(X_train, y_train)
print("\n[+] Done: Model Training Complete")

print("\n[****] Prediction Started")
# prediction on the training data
prediction_test_train = model.predict(X_train)
#prediction on testing data.
prediction_test = model.predict(X_test)
print("\n[+] DONE: Prediction")

from sklearn import metrics
#Print the prediction accuracy
#First check the accuracy on training data.
print ("[+] Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset.
print ("[+] Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

#eature importances
importances = list(model.feature_importances_)
print("[+] List of FEATURES: ",importances)

#importances = list(model.feature_importances_)
#print("[+]List of IMP FEATURES: ",importances)
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print("\n[****] Listing IMP Features\n",feature_imp)


#SAVE MODEL
if not os.path.exists(PATH+"RF_FILES/RForest_model_1.pkl"):
    print("[****] SAVING MODEL")
    import pickle
    filename = PATH+"RF_FILES/RForest_model_1.pkl"
    pickle.dump(model,open(filename,'wb'))
    print("\n[+] DONE: Model Saved")
else:
    print("\n[+] EXISTS: MODEL")