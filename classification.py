from flaml import autogen
from flaml.automl import AutoML
from sklearn.datasets import load_iris
automl=AutoML()
automl_settings={
    "time_budget":10,#train model in 10 sec
    "metric":'accuracy',
    "task":'classification',
    "log_file_name":"iris.log",#see all training details
}
# Load a dataset (for example)
X,y=load_iris(return_X_y=True,as_frame=True)#in numpy format

#split data into training and testing stes for a more realistic scenario
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#optional---------------------------------------- use this line----------- X_train,y_train=load_iris(return_X_y=True,as_frame=True)#in numpy format

#train the label input data 

# 4. Train the model
print("--- Starting AutoML Training ---")
automl.fit(X_train=X_train,y_train=y_train,**automl_settings)
print("--- AutoML Training Finished ---\n")


#display the results of the training
print("BEST MODEL FOUND")
print(f"Best learner: {automl.best_estimator}")
print(f"Best Configuration:{automl.best_config}")
print(f"Best accuracy on validation data:{1-automl.best_loss:.4f}\n")

#predict
print("AUTOML PREDICT")
print(automl.predict_proba(X_train))


print("--- Making Predictions on Test Data ---")

# Get the final predicted class labels (e.g., 0, 1, or 2)
predictions = automl.predict(X_test)
print("Predicted Classes (first 5):")
print(predictions[:5])

# Get the predicted probabilities for each class
probabilities = automl.predict_proba(X_test)
print("\nPredicted Probabilities (first 5):")
print(probabilities[:5])


#export thebest model
print("HRERE THE MODEL")
print(automl.model)