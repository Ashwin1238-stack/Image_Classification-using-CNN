# 📊 CIFAR-10 Image Classification using CNN

### A Deep Learning Project on Image Recognition

---

## 📌 Project Overview

This project demonstrates image classification on the **CIFAR-10 dataset** using a **Convolutional Neural Network (CNN)**.  
The CIFAR-10 dataset consists of 60,000 images across 10 distinct categories, such as airplanes, automobiles, birds, cats, and more.

---

## 📈 Performance Visualization

### ✅ Accuracy Over Epochs

```text
sns.lineplot(history.history['accuracy'], label='Train Accuracy', marker='o')
sns.lineplot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.legend()
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
📉 Loss Over Epochs
text
Copy
Edit
sns.lineplot(history.history['loss'], label='Train Loss')
sns.lineplot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
🔮 Sample Image Prediction
text
Copy
Edit
def predict_image(index):
    image = X_test[index]
    true_label = int(np.argmax(Y_test[index]))
    img_input = np.expand_dims(image, axis=0)
    prediction = model.predict(img_input)
    predicted_class = int(np.argmax(prediction))

    plt.figure(figsize=(2,2))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Predicted: {class_names[predicted_class]} | Actual: {class_names[true_label]}')
    plt.show()
Example:

text
Copy
Edit
predict_image(6)
Output:

text
Copy
Edit
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step
🚀 Technologies Used
Python

TensorFlow / Keras

Matplotlib

Seaborn

NumPy

CIFAR-10 Dataset

📂 Project Structure
bash
Copy
Edit
├── data/                # CIFAR-10 dataset (via keras.datasets)
├── model/               # CNN model and training scripts
├── results/             # Accuracy & Loss visualization plots
├── README.md            # Project documentation
└── main.ipynb           # Jupyter Notebook implementation
🎯 Results Summary
Achieved strong accuracy on both training and validation datasets.

Clear visualizations show model performance trends.

Individual sample predictions demonstrate model reliability.

👤 Author
Ashwin Kumar
Data Analyst | AI Enthusiast | Deep Learning Explorer
