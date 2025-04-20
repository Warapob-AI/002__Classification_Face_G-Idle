from sklearn.metrics import accuracy_score, classification_report

def classification(y_test, y_pred):
    print(f'ความแม่นยำ : {accuracy_score(y_test, y_pred) * 100:.2f}%')
    print(classification_report(y_test, y_pred, target_names=["Minnie", "Miyeon", "Shuhua", "Yuqi", "Soyeon"]))