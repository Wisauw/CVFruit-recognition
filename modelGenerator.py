import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir el generador de datos
datagen = ImageDataGenerator(
    rescale=1.0/255.0, 
    validation_split=0.2  # Usa 20% de los datos para validación
)

# Cargar los datos de entrenamiento
train_generator = datagen.flow_from_directory(
    'D:\\Downloads\\fruits-360_dataset\\fruits-360\\Training',
    target_size=(100, 100),  # Tamaño al que se redimensionarán las imágenes
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Cargar los datos de validación
validation_generator = datagen.flow_from_directory(
    'D:\\Downloads\\fruits-360_dataset\\fruits-360\\Training',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Guardar el modelo entrenado
model.save('modelo_frutas.h5')
