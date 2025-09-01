import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., np.newaxis]  
x_test = x_test[..., np.newaxis]
archive_path = r"C:\Users\averm\Documents\projectfile\venv\archive"

archive_ds = image_dataset_from_directory(
    archive_path,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(28,28),
    batch_size=None  
)

archive_images = []
archive_labels = []
for img, label in archive_ds:
    archive_images.append(img.numpy())
    archive_labels.append(label.numpy())

archive_images = np.array(archive_images) / 255.0
archive_labels = np.array(archive_labels)

archive_images = np.squeeze(archive_images)
if archive_images.ndim == 3:
    archive_images = archive_images[..., np.newaxis]
X_all = np.concatenate([x_train, archive_images], axis=0)
y_all = np.concatenate([y_train, archive_labels], axis=0)

my_digits_path = r"C:\Users\averm\Documents\projectfile\venv\digits"

my_digits_ds = image_dataset_from_directory(
    my_digits_path,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(28,28),
    batch_size=None
)

my_images = []
my_labels = []
for img, label in my_digits_ds:
    my_images.append(img.numpy())
    my_labels.append(label.numpy())

my_images = np.array(my_images) / 255.0
my_labels = np.array(my_labels)

my_images = np.squeeze(my_images)
if my_images.ndim == 3:
    my_images = my_images[..., np.newaxis]

X_all = np.concatenate([x_train, archive_images, my_images], axis=0)
y_all = np.concatenate([y_train, archive_labels, my_labels], axis=0)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
)

model = load_model("mnist_cnn_model.h5")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,       # small rotations
    width_shift_range=0.1,   # horizontal shift
    height_shift_range=0.1,  # vertical shift
    shear_range=0.1,         # slanting
    zoom_range=0.1           # zoom in/out
)

# 
datagen.fit(X_train)

# train with augmented data
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=10,   # train a bit longer since augmentation makes it harder
    steps_per_epoch=len(X_train) // 64
)

model.save("mnist_archive_combined.h5")
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')   
    img = img.resize((28,28))
    img_array = np.array(img) / 255.0
    img_array = 1 - img_array                   
    img_array = img_array[np.newaxis, ..., np.newaxis]  
    return img_array

image_folder = r"C:\Users\averm\Documents\projectfile\venv\my_digits"
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

for file in image_files:
    path = os.path.join(image_folder, file)
    img_array = preprocess_image(path)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    print(f"{file}: Predicted digit = {predicted_label}")

    plt.imshow(img_array[0,:,:,0], cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()