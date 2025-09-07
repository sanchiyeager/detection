from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import svm, metrics
import pandas as pd


# Load the data
data = pd.read_csv('main.csv')
x = data[['sleep_interval','appetite','going_out','friendship_status','family_relationships','relationship','hobbies_and_interests','self_perception','bullying_experience','social_media_impact','academic_pressure','intoxication','_and_ambitions','age','grade']].values
y = data[['depressionLevel']].values


le = LabelEncoder()
for i in range(x.shape[1]):
    x[:,i]= le.fit_transform(x[:, i])

y=le.fit_transform(y.ravel())
print(y[:20])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.45)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

smote_test = SMOTE()
x_test, y_test = smote_test.fit_resample(x_test, y_test)


class Models:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def svm_model(self):
        classify = svm.SVC()
        classify.fit(self.x_train, self.y_train)
        predict = classify.predict(self.x_test)
        accuracy = metrics.accuracy_score(predict, self.y_test)
        score = metrics.f1_score(self.y_test, predict, average='weighted')
        precision = metrics.precision_score(self.y_test, predict, average='weighted')
        recall = metrics.recall_score(self.y_test, predict, average='weighted')
        return accuracy, score, precision, recall

if __name__ == '__main__':

     model = Models(x_train, x_test, y_train, y_test)
     svm_acc, svm_f1, svm_pre, svm_rec = model.svm_model()

    # Output results for SVM model
     print(f"SVM Model: \n1)Accuracy: {svm_acc * 100:.2f}% \n2)F1_score: {svm_f1 * 100:.2f}% \n3)Precision: {svm_pre * 100:.2f}% \n4)Recall: {svm_rec * 100:.2f}%")
