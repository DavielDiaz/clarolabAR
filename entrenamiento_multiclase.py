import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import subprocess

# ----------------------------------------
# 1. Parámetros
# ----------------------------------------

IMG_SIZE = (128, 128)
DATASET_DIR = "Dataset"
EXPORT_DIR = "ModeloMulticlase"
CLASES = ["AtariAsteroids", "GeniusOldMouse", "MagicBrainCalculator"]  # Clases reales
NEGATIVO = "negativos"  # Carpeta con imágenes sin objetos

# ----------------------------------------
# 2. Carga de datos
# ----------------------------------------

images = []
labels = []

# Para cada clase, cargamos las imágenes y generamos etiquetas con bounding box
for idx, clase in enumerate(CLASES):
    path = os.path.join(DATASET_DIR, clase)
    for nombre in os.listdir(path):
        if nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(path, nombre)
            img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            img_array = np.array(img) / 255.0

            label = []
            for i in range(len(CLASES)):
                if i == idx:
                    # Objeto presente con bbox centrado y tamaño relativo (ficticio)
                    label.extend([1.0, 0.5, 0.5, 0.4, 0.4])
                else:
                    # Resto de clases no presentes
                    label.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            images.append(img_array)
            labels.append(label)

# Cargar negativos (ningún objeto presente)
neg_path = os.path.join(DATASET_DIR, NEGATIVO)
for nombre in os.listdir(neg_path):
    if nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(neg_path, nombre)
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img) / 255.0

        # Todo 0s = sin objetos
        label = [0.0] * (5 * len(CLASES))
        images.append(img_array)
        labels.append(label)

# Convertir a arrays de numpy
images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

print(f"✅ Datos cargados: {images.shape[0]} imágenes")
print(f"   Formato imagen: {images.shape[1:]} - Formato etiquetas: {labels.shape[1:]}")

# ----------------------------------------
# 3. Visualización rápida (opcional)
# ----------------------------------------

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
# 4. División en train y val
# ----------------------------------------

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# ----------------------------------------
# 5. Dataset TensorFlow + Aumento
# ----------------------------------------

# Aumentos solo en entrenamiento
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
# 6. Función de pérdida personalizada
# ----------------------------------------

def custom_multiclass_loss(y_true, y_pred):
    total_loss = 0.0
    for i in range(len(CLASES)):
        # Para cada clase, extraer su sección de 5 valores
        y_true_c = y_true[:, i*5:(i+1)*5]
        y_pred_c = y_pred[:, i*5:(i+1)*5]

        y_true_prob = y_true_c[:, 0]
        y_pred_prob = y_pred_c[:, 0]
        y_true_box = y_true_c[:, 1:]
        y_pred_box = y_pred_c[:, 1:]

        # Pérdida de clasificación
        classification_loss = tf.keras.losses.binary_crossentropy(y_true_prob, y_pred_prob)

        # Pérdida de bounding box solo si hay objeto
        object_mask = tf.expand_dims(y_true_prob, axis=-1)
        bbox_loss = tf.reduce_sum(tf.square(y_true_box - y_pred_box) * object_mask, axis=-1)

        total_loss += classification_loss + bbox_loss

    return total_loss

# ----------------------------------------
# 7. Modelo CNN
# ----------------------------------------

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
# 8. Entrenamiento
# ----------------------------------------

EPOCHS = 60
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ----------------------------------------
# 9. Exportar a TensorFlow.js
# ----------------------------------------

saved_model_path = "modelo_temp_saved"
tfjs_output_path = EXPORT_DIR

# Exportar a formato SavedModel
model.export(saved_model_path)

# Convertir a TensorFlow.js usando subprocess
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
