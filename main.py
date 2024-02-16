from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

filename = "Energy Meter.csv"
names = ["Voltage","Current","Power","Class"]
dataset = read_csv(filename,names=names)

array = dataset.values
x = array[:,0:3]
y = array[:,3]
X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# model = SVC(gamma='auto') Support Vector Machine ML classifier shows only 98% Accuracy but LR has more
model = LogisticRegression(solver='liblinear',multi_class='ovr')
model.fit(X_train, y_train)

results = model.score(X_validation,y_validation) #Has a 100 percent model accuracy! with the LR Model
print(f"{results*100} % accuracy")

while True:
    x1 = float(input("Input your voltage from the volt meter (input in decimals for accuracy or real numbers your choice) "))
    x2 = float(input("Enter your current from the amp meter (input in decimals for accuracy or real numbers your choice)"))
    x3 = float(input("Enter your power consuption of your house currently (input in decimals for accuracy or real numbers your choice)"))

    predictions = [[x1,x2,x3]]
    predictions = model.predict(predictions)
    print(predictions)

    check = input("Do you want to continue your test (type yes or no)")
    if check=="yes".lower():
        continue
    if check=="no".lower():
        break