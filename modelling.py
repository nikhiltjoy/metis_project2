from sklearn import preprocessing
import sklearn
import pandas as pd
import numpy as np


def convert_to_categories(df, columns):
    les = {}
    for column in columns:
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])
        les[column] = le
    return (df, les)


def decode_categories(df, columns, les):
    for column in columns:
        df[column] = les[column].inverse_transform(df[column])
    return df


def k_model_bank(df, k):
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=3)
    cols = [col for col in df.columns if col not in ['y', 'poutcome']]
    knn_model = KNeighborsClassifier(n_neighbors=k).fit(
        train_set[cols], train_set['y'])
    pred_y = knn_model.predict(test_set[cols])
    f1 = sklearn.metrics.f1_score(test_set['y'], pred_y, average=None)
    ps = sklearn.metrics.precision_score(test_set['y'], pred_y, average=None)
    rs = sklearn.metrics.recall_score(test_set['y'], pred_y, average=None)
    acc_s = sklearn.metrics.accuracy_score(test_set['y'], pred_y)
    return {'k': k, 'f1': f1, 'precision': ps, 'accuracy': acc_s, 'recall': rs}


bank_data = pd.read_csv(open('bank.csv', 'r'),
                        delimiter=';').replace(('yes', 'no', 'unknown'), (1, 0, None))
columns = ['job', 'marital', 'education', 'month', 'contact', 'poutcome']
bank_data, les = convert_to_categories(bank_data, columns)
print bank_data.head()
print max(map(lambda x: k_model_bank(bank_data, x), range(10, 31)), key=itemgetter('accuracy'))
bank_data = decode_categories(bank_data, columns, les)
print bank_data.head()
