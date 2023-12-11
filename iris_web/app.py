from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")



import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense



iris = load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание нейронной сети
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test))





@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    def predict_flower_species(sepal_length, sepal_width, petal_length, petal_width):
        # Нормализация введенных данных
        input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

        # Предсказание класса цветка
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions[0])

        # Определение наименования цветка на основе предсказанного класса
        flower_species = iris.target_names[predicted_class]

        return flower_species
    


    predicted_species = predict_flower_species(sepal_length, sepal_width, petal_length, petal_width)

    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": f"Предсказанное наименование цветка: {predicted_species}"})