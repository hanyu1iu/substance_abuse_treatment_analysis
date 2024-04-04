from encode_join import *
from tqdm.notebook import tqdm

train, test = pickle.load(open('split.p', 'rb'))
batch_size = 10000
num_batch = int(len(train)/batch_size) + 1
np.random.seed(47)
batches = np.arange(num_batch)

from sklearn.linear_model import SGDClassifier
alpha = 1e-3
tol = 0.01
loss = 'hinge'

def train_clf(clf, patience, _iters):
    best_acc = 0
    train_batch_scores = []
    for epoch in tqdm(range(3)):
        print('epoch {}'.format(epoch))
        running_accuracy = 0
        no_improvement = 0
        np.random.shuffle(batches)
        for i in tqdm(range(num_batch)):
            _i = batches[i]
            ind = train[_i*batch_size:(_i+1)*batch_size]
            X_this_batch, y_this_batch = load_batch(ind)
            for j in range(_iters):
                if epoch==0 and i==0 and j==0:
                    clf.partial_fit(X_this_batch, y_this_batch, classes=np.arange(2))
                else:
                    clf.partial_fit(X_this_batch, y_this_batch)
            this_accuracy = clf.score(X_this_batch, y_this_batch)
            train_batch_scores.append(this_accuracy)
            running_accuracy += this_accuracy
            running_acc_avg = running_accuracy/(i+1)
            print('batch {}: running acc={:.5f}, batch acc={:.5f}'.format(i, running_acc_avg, this_accuracy))
            if running_acc_avg > best_acc:
                no_improvement = 0
                best_acc = running_acc_avg
            else:
                no_improvement += 1
            if no_improvement > patience:
                print('Training ended early.')
                return clf, train_batch_scores
    return clf, train_batch_scores
            

L1_SVM_clf = SGDClassifier(penalty='l1', alpha=alpha, loss=loss, tol=tol, warm_start=True)
L1_SVM_clf, L1_training_scores = train_clf(L1_SVM_clf, patience=100, _iters=10)

L2_SVM_clf = SGDClassifier(penalty='l2', alpha=1e-3, loss=loss, tol=tol, warm_start=True)
L2_SVM_clf, L2_training_scores = train_clf(L2_SVM_clf, patience=200, _iters=20)

from joblib import dump, load
dump(L1_SVM_clf, 'L1_SVM_clf.joblib')
dump(L2_SVM_clf, 'L2_SVM_clf.joblib') 