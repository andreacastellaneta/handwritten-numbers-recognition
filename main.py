import numpy as np
import matplotlib.pyplot as plt
from multiclass_nn import img_recognition_mdl, img_recognition_test

"""

MNIST set contains a collection of 70,000, 28 x 28 images of handwritten digits from 0 to 9. 
The dataset is already divided into training and testing sets

"""
# Loading the MNIST dataset
from keras.datasets import mnist

# Loading training and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('Creating the NN...')
model = img_recognition_mdl(X_train, y_train, 40)

# Evaluation
y_preds, acc = img_recognition_test(model, X_test, y_test)
print(f'Accuracy: {acc}')

# Plotting some results on test set
fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(X_test.shape[0])

    # Select rows corresponding to the random indices and reshape the image
    X_random_reshaped = X_test[random_index].reshape((28, 28))

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Display the label above the image
    ax.set_title(f"{y_test[random_index]},{y_preds[random_index]}", fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, y_hat", fontsize=13)

plt.show()
