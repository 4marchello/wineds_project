# МОДУЛЬ: train_model. Навчання та метрики.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def navchaty(xtrain,ytrain):
    model=LogisticRegression(max_iter=1000,random_state=42)
    model.fit(xtrain,ytrain)
    return model

def ocinyty(model,xtest,ytest):
    prognoz=model.predict(xtest)
    metryky={
        "accuracy":accuracy_score(ytest,prognoz),
        "precision":precision_score(ytest,prognoz,average="macro"),
        "recall":recall_score(ytest,prognoz,average="macro"),
        "f1":f1_score(ytest,prognoz,average="macro"),
        "matrix":confusion_matrix(ytest,prognoz)
    }
    return metryky