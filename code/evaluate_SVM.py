from encode_join import *
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

L1_SVM_clf = load('L1_SVM_clf.joblib')
L2_SVM_clf = load('L2_SVM_clf.joblib')

def full_join(ind):
    batch = df.loc[ind]
    batch = pd.merge(batch, state_data, how="left", left_on=["STFIPS", "DISYR"], right_on=["fips", "Year"])
    batch = batch.drop(labels=["fips", "Year"], axis=1)
    batch = pd.merge(batch, CBSA_data, how="left", left_on=["CBSA", "DISYR"], right_on=["CBSA", "Year"])
    batch = batch.drop(labels=[ "Year"], axis=1)
    batch = pd.merge(batch, facs_data, how="left", left_on=["STFIPS", "DISYR"], right_on=["state_fips", "year"])
    batch = batch.drop(labels=["state_fips", "year"], axis=1)
    return batch

dummy = full_join(np.arange(5))
features = list(dummy.columns[4:])
treatment_features = features[:249]
state_features = features[249:743]
CBSA_features = features[743:1237]
facs_features = features[1237:]

def get_all_y(ind):
    y = df.loc[ind].REASON.values
    return np.int16(y==1)

def evaluate_all(clf, ind):
    y_score = []
    y_pred = []

    batch_size = 10000
    num_batch = int(len(ind)/batch_size) + 1
    for i in tqdm(range(num_batch)):
        _ind = ind[i*batch_size:(i+1)*batch_size]
        X_this_batch, _ = load_batch(_ind)
        y_score.append(clf.decision_function(X_this_batch))
        y_pred.append(clf.predict(X_this_batch))

    cat = lambda _list: np.concatenate(_list)
    return cat(y_score), cat(y_pred)

y_train, y_test = get_all_y(train), get_all_y(test)

y_score_test_l1, y_pred_test_l1 = evaluate_all(L1_SVM_clf, test)
y_score_test_l2, y_pred_test_l2 = evaluate_all(L2_SVM_clf, test)

accuracy_score(y_test, y_pred_test_l2)
f1_score(y_test, y_pred_test_l2)
roc_auc_score(y_test, y_pred_test_l2)

def top_k(clf, k):
    _ind = np.argsort(np.abs(clf.coef_[0]))[::-1][:k]
    _feats = np.array(features)[_ind]
    coeffs = clf.coef_[0][_ind]
    return _feats, coeffs

l1_ticks, l1_heights = top_k(L1_SVM_clf, 10)
l2_ticks, l2_heights = top_k(L2_SVM_clf, 10)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
axs[0].title.set_text('SVM (l1) top 10 features')
axs[0].bar(l1_ticks, l1_heights)
axs[0].tick_params(labelrotation=90)
axs[1].title.set_text('SVM (l2) top 10 features')
axs[1].bar(l2_ticks, l2_heights)
axs[1].tick_params(labelrotation=90)
plt.show()