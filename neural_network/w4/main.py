#Avaya Khatri

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import img_to_array

# 1. LOAD CIFAR-10 DATASET
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("Dataset loaded successfully.")
print("Test image shape:", x_test.shape)

# 2. LOAD PRETRAINED RESNET-152 MODEL
print("\nLoading pretrained ResNet-152 model...")
model = ResNet152(weights='imagenet')
print("Model loaded successfully.")

# 3. SELECT 10 RANDOM TEST IMAGES
num_images = 10
random_indices = random.sample(range(len(x_test)), num_images)

selected_images = x_test[random_indices]
selected_labels = y_test[random_indices]

# 4. PREPROCESS IMAGES FOR RESNET-152
processed_images = []

for img in selected_images:
    resized_img = tf.image.resize(img, (224, 224)).numpy()
    resized_img = img_to_array(resized_img)
    processed_images.append(resized_img)

processed_images = np.array(processed_images)
processed_images = preprocess_input(processed_images)

# 5. MAKE PREDICTIONS
print("\nMaking predictions on 10 random images...")
predictions = model.predict(processed_images, verbose=0)
decoded_predictions = decode_predictions(predictions, top=3)

# 6. PRINT RESULTS IN TERMINAL
print("\n========== PREDICTION RESULTS ==========\n")

for i in range(num_images):
    actual_label = class_names[selected_labels[i][0]]
    top_pred = decoded_predictions[i][0]

    pred_class_name = top_pred[1]
    pred_confidence = top_pred[2] * 100

    print(f"Image {i+1}")
    print(f"Actual Label    : {actual_label}")
    print(f"Top Prediction  : {pred_class_name}")
    print(f"Confidence      : {pred_confidence:.2f}%")
    print("Top 3 Predictions:")
    for rank, pred in enumerate(decoded_predictions[i], start=1):
        print(f"  {rank}. {pred[1]} ({pred[2] * 100:.2f}%)")
    print("-" * 45)

# 7. DISPLAY 10 IMAGES WITH ACTUAL/PREDICTED LABELS
plt.figure(figsize=(18, 8))

for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(selected_images[i].astype("uint8"))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    actual_label = class_names[selected_labels[i][0]]
    predicted_label = decoded_predictions[i][0][1]
    confidence = decoded_predictions[i][0][2] * 100

    plt.title(
        f"Actual: {actual_label}\nPred: {predicted_label}\n{confidence:.1f}%",
        fontsize=10
    )

plt.suptitle("ResNet-152 Predictions on 10 Random CIFAR-10 Images", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()