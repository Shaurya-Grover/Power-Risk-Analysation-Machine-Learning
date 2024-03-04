from tkinter import *
from tkinter import messagebox
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
filename = "Energy_Meter.csv"
names = ["Voltage", "Current", "Power", "Class"]
dataset = read_csv(filename, names=names)

# Prepare data
array = dataset.values
x = array[:, 0:3]
y = array[:, 3]
X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# Train logistic regression model
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)

# Create Tkinter window
root = Tk()
root.title("Energy Meter Prediction")
root.geometry("1280x720")

def predict_status():
    # Get input values
    voltage = float(entry_voltage.get())
    current = float(entry_current.get())
    power = float(entry_power.get())

    # Predict
    prediction = model.predict([[voltage, current, power]])

    # Display prediction
    messagebox.showinfo("Prediction", f"The status is: {prediction[0]}")

# Create input fields
label_voltage = Label(root, text="Voltage:")
label_voltage.grid(row=0, column=0, pady=10)
entry_voltage = Entry(root, width=20)
entry_voltage.grid(row=0, column=1, pady=10)

label_current = Label(root, text="Current:")
label_current.grid(row=1, column=0, pady=10)
entry_current = Entry(root, width=20)
entry_current.grid(row=1, column=1, pady=10)

label_power = Label(root, text="Power:")
label_power.grid(row=2, column=0, pady=10)
entry_power = Entry(root, width=20)
entry_power.grid(row=2, column=1, pady=10)

# Create predict button
predict_button = Button(root, text="Predict", command=predict_status, width=20)
predict_button.grid(row=3, column=0, columnspan=2, pady=20)

root.mainloop()
