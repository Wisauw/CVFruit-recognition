import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

model = load_model('modelo_frutas.h5')

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    'D:\\Downloads\\fruits-360_dataset\\fruits-360\\Test',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  #
)


Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


cm = confusion_matrix(y_true, y_pred)


cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


plt.figure(figsize=(20, 20))  
sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.show()

report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)
