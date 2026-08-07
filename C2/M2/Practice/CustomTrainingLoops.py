import tensorflow as tf
import numpy as np


#1. Define the model(Custom)

class LinearModel (tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call (self, inputs):
        return self.dense(inputs)

model = LinearModel()

#2. Prep the synthetic data

X_train = np.random.rand(100, 1).astype(np.float32)
y_train = 3*X_train + 2 + 0.1* np.random.randn(100,1).astype(np.float32)

#2.1 Break the data into batches

dataset = tf.keras.Dataset.from_tensor_slices((X_train, y_train)).batch(10)

#3. Define custom loss and optimezer functions

loss_fn = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

#4. Training models

epochs = 100

for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)

        #Compute gradients
        gradients = tape.gradients(loss, model.trainable_variables)
        #Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
