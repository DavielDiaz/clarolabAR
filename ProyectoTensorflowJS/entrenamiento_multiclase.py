# entrenamiento.py
# Entrena un modelo CNN multiclase para detección de objetos (presencia + bounding boxes)
# y lo exporta a formato TensorFlow.js para uso en navegador.

# Requiere TensorFlow, PIL, NumPy, Matplotlib y scikit-learn instalados.
## Asegúrate de tener pip actualizado
# Comando de terminal: python -m pip install --upgrade pip

## Instala las librerías necesarias
# Comando de terminal: pip install tensorflow numpy pillow scikit-learn matplotlib

import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import subprocess

# ----------------------------------------
# 1. PARÁMETROS GENERALES
# ----------------------------------------

IMG_SIZE = (128, 128)
DATASET_DIR = "Dataset"
EXPORT_DIR = "ModeloMulticlase"

# Lista de clases a detectar (nombres deben coincidir con las subcarpetas en el dataset)
# En caso de querer añadir más clases, simplemente agrega sus nombres aquí.
CLASES = ["AtariAsteroids", "GeniusOldMouse", "MagicBrainCalculator"]
NEGATIVO = "negativos"  # Carpeta con imágenes sin objetos (clase nula)

# ----------------------------------------
# 2. CARGA Y PREPARACIÓN DE DATOS
# ----------------------------------------

images = []
labels = []

# Se recorre cada clase para cargar las imágenes y generar etiquetas.
# Cada etiqueta contiene: [probabilidad, x_center, y_center, width, height] por clase (formato similar a YOLO).
for idx, clase in enumerate(CLASES):
    path = os.path.join(DATASET_DIR, clase)
    for nombre in os.listdir(path):
        if nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(path, nombre)
            img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            img_array = np.array(img) / 255.0

            # Construimos una etiqueta codificada: para la clase presente se añade prob=1 y bbox ficticio centrado
            label = []
            for i in range(len(CLASES)):
                if i == idx:
                    label.extend([1.0, 0.5, 0.5, 0.4, 0.4])  # bbox fijo en centro
                else:
                    label.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # no presente

            images.append(img_array)
            labels.append(label)

# Procesamos las imágenes "negativas": no contienen ningún objeto de interés
neg_path = os.path.join(DATASET_DIR, NEGATIVO)
for nombre in os.listdir(neg_path):
    if nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(neg_path, nombre)
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img) / 255.0

        # Etiqueta completamente nula para clase negativa
        label = [0.0] * (5 * len(CLASES))
        images.append(img_array)
        labels.append(label)

# Convertimos a tensores numpy para alimentar a TensorFlow
images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

print(f"✅ Datos cargados: {images.shape[0]} imágenes")
print(f"   Formato imagen: {images.shape[1:]} - Formato etiquetas: {labels.shape[1:]}")

# ----------------------------------------
# 3. VISUALIZACIÓN (opcional)
# ----------------------------------------

# Muestra una vista previa de 5 imágenes y sus etiquetas (clase detectada)
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    etiqueta = labels[i]
    texto = " / ".join(
        [CLASES[j] for j in range(len(CLASES)) if etiqueta[j * 5] == 1.0]
    ) or "Negativo"
    plt.title(texto, fontsize=8)
    plt.axis("off")
plt.savefig("preview_multiclase.png")

# ----------------------------------------
# 4. DIVISIÓN EN TRAIN Y VALIDACIÓN
# ----------------------------------------

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# ----------------------------------------
# 5. DATASET DE TENSORFLOW + AUMENTO
# ----------------------------------------

# Aumento de datos solo para entrenamiento: mejora la capacidad de generalización
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.shuffle(100).batch(16).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)

# ----------------------------------------
# 6. FUNCIÓN DE PÉRDIDA PERSONALIZADA
# ----------------------------------------

# Esta función combina:
# - pérdida de clasificación (probabilidad de presencia de cada clase)
# - pérdida de localización (bounding box), activada solo cuando hay objeto presente

def custom_multiclass_loss(y_true, y_pred):
    total_loss = 0.0
    for i in range(len(CLASES)):
        y_true_c = y_true[:, i*5:(i+1)*5]
        y_pred_c = y_pred[:, i*5:(i+1)*5]

        y_true_prob = y_true_c[:, 0]
        y_pred_prob = y_pred_c[:, 0]
        y_true_box = y_true_c[:, 1:]
        y_pred_box = y_pred_c[:, 1:]

        classification_loss = tf.keras.losses.binary_crossentropy(y_true_prob, y_pred_prob)

        object_mask = tf.expand_dims(y_true_prob, axis=-1)
        bbox_loss = tf.reduce_sum(tf.square(y_true_box - y_pred_box) * object_mask, axis=-1)

        total_loss += classification_loss + bbox_loss

    return total_loss

# ----------------------------------------
# 7. MODELO CNN SIMPLE
# ----------------------------------------

# Arquitectura sencilla para ejecutarse en navegador:
# - 3 bloques Conv + MaxPool
# - Capa densa con dropout
# - Capa final de salida con sigmoid para multilabel + regresión

inputs = tf.keras.Input(shape=(128, 128, 3))
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(CLASES) * 5, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss=custom_multiclass_loss)
model.summary()

# ----------------------------------------
# 8. ENTRENAMIENTO DEL MODELO
# ----------------------------------------

EPOCHS = 60
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ----------------------------------------
# 9. EXPORTACIÓN A TENSORFLOW.JS
# ----------------------------------------

# Se guarda el modelo como SavedModel y luego se convierte a formato TensorFlow.js con subprocess

saved_model_path = "modelo_temp_saved"
tfjs_output_path = EXPORT_DIR

model.export(saved_model_path)

comando = [
    "tensorflowjs_converter",
    "--input_format", "tf_saved_model",
    "--output_format", "tfjs_graph_model",
    saved_model_path,
    tfjs_output_path
]

try:
    subprocess.run(comando, check=True)
    print("✅ Modelo exportado a TensorFlow.js en:", tfjs_output_path)
except subprocess.CalledProcessError as e:
    print("❌ Error en la conversión TFJS:", e.stderr)
