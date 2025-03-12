import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from keras import models
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

#hdf5 example of use

def cm_plot(y, yp):
    '''
    y: real label
    yp: predicted label
    '''
    classes=['EQ','EP','SS']
    cm = confusion_matrix(y, yp)  
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()

    for x in range(len(cm)):  
        for y in range(len(cm)):
            if cm[x, y] > 10:
                plt.annotate(cm[x, y], xy=(y, x), verticalalignment='center', horizontalalignment='center', color='w')
            else:
                plt.annotate(cm[x, y], xy=(y, x), verticalalignment='center', horizontalalignment='center')
            
    tick_marks =np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.show()

def roc(y, predict):
    '''
    y: real label
    predict: predicted value
    '''
    results = []
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], predict[:, i], drop_intermediate=True)
        roc_auc = auc(fpr, tpr)
        results.append([fpr, tpr, roc_auc])
    col = ['blue', 'red', 'green']
    classes = ['EQ', 'EP', 'SS']
    for i, (fpr, tpr, rocauc) in enumerate(results):
        plt.plot(fpr, tpr, color=col[i], lw=2,label=f'{classes[i]} Auc:%0.2f' % rocauc)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Reference line')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.legend(loc='lower right')
    plt.show()

#demo: There are 20 separate of earthquakes,explosions and collapses
#label: earquake 0; explosion 1; collapse 2

ternary_test =np.load('demo.npz')
x=[ternary_test[feature] for feature in ternary_test if feature!='type']
y=ternary_test['type']

print(f"The samples number: {y.shape}")
model = models.load_model('ternary_model.hdf5')
test_loss, test_acc = model.evaluate(x, y)
predict = model.predict(x)
pre = np.argmax(predict, axis=1)

print(f"Test accuracy: {test_acc:.3f}\n"
      "earthquake explosion collapse\n"
      f"Precision: {[round(i,3) for i in precision_score(y, pre, average=None)]}\n"
      f"Recall: {[round(i,3) for i in recall_score(y, pre, average=None)]}\n"
      f"F1-score: {[round(i,3) for i in f1_score(y, pre, average=None)]}\n"
      f"macroF1-score: {f1_score(y, pre, average='macro'):.3f}\n")

cm_plot(y, pre)
roc(y, predict)