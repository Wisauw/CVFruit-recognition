import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('modelo_frutas.h5')

datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(
    'D:\\Downloads\\fruits-360_dataset\\fruits-360\\Training',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)

class_names = list(train_generator.class_indices.keys())


def predict_fruit(image, model):
    img_resized = cv2.resize(image, (100, 100))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_names[class_idx]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    fruit_name = predict_fruit(frame, model)
    
    cv2.putText(frame, fruit_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Reconocimiento de Frutas', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
