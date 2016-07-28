from sklearn import preprocessing
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# def convert_to_categories(df, columns):
#     les = {}
#     for column in columns:
#         le = preprocessing.LabelEncoder()
#         df[column] = le.fit_transform(df[column])
#         les[column] = le
#     return (df, les)


def decode_categories(df, columns, les):
    for column in columns:
        df[column] = les[column].inverse_transform(df[column])
    return df


def k_model_bank(df, k):
    train_set, test_set = train_test_split(
        df.dropna(how='any'), test_size=0.4, random_state=3)
    cols = [col for col in df.columns if col not in ['y']]
#     cols = ['marital', 'contact', 'previous']
#     print cols
    knn_model = KNeighborsClassifier(n_neighbors=k).fit(
        train_set[cols], train_set['y'])
    pred_y = knn_model.predict(test_set[cols])
    f1 = sklearn.metrics.f1_score(test_set['y'], pred_y, average=None)
    ps = sklearn.metrics.precision_score(test_set['y'], pred_y, average=None)
    rs = sklearn.metrics.recall_score(test_set['y'], pred_y, average=None)
    acc_s = sklearn.metrics.accuracy_score(test_set['y'], pred_y)
    conf_m = sklearn.metrics.confusion_matrix(
        test_set['y'], pred_y, labels=[0, 1])
    return {'k': k, 'f1': f1, 'precision': ps, 'accuracy': acc_s, 'recall': rs, 'matrix': conf_m}


df = pd.read_csv(open('../project_2/data/bank/bank.csv', 'r'),
                 delimiter=';').replace(('yes', 'no', 'unknown'), (1, 0, None))
bank_data = df[['balance', 'pdays', 'housing', 'age', 'y']]
cat_dummies = pd.get_dummies(df['poutcome'].fillna(
    'other'), dummy_na=False, prefix='poutcome')
new_data = pd.concat([bank_data, cat_dummies], axis=1, join='inner')
mean = np.mean(new_data['pdays'].replace((-1,), (None,)).dropna())
new_data['new_call'] = bank_data['pdays'].apply(lambda x: x < 0)
new_data['pdays'] = new_data['pdays'].apply(lambda x: (x if x >= 0 else mean))
print new_data.head()

# cols = [col for col in bank_data.columns if col not in ['y']]
# sel = VarianceThreshold(threshold=0.16).fit_transform(bank_data[cols])
# print sel
# print bank_data.head()

m = max(map(lambda x: k_model_bank(bank_data, x),
            range(1, 50)), key=itemgetter('accuracy'))
print m
# print m['matrix']
# bank_data = decode_categories(bank_data, columns, les)
