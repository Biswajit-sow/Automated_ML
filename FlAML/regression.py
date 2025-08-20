###regression problem
from flaml import autogen
from flaml.automl import AutoML
from sklearn.datasets import fetch_california_housing

automl=AutoML()
automl_settings={
    "time_budget":10,
    "metric":'r2',
    "task":'regression',
    "log_file_name":"california.log",
}
X_train,y_train=fetch_california_housing(return_X_y=True)
#train the model
automl.fit(X_train=X_train,y_train=y_train,**automl_settings)

print("BEST MODEL FOUND")
print(f"Best learner: {automl.best_estimator}")
print(f"Best Configuration:{automl.best_config}")
print(f"Best accuracy on validation data:{1-automl.best_loss:.4f}\n")
print(automl.model)

#PREDICT
print(automl.predict(X_train))

