import time
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
scale = 11000
x_train, y_train = x_train[:scale], y_train[:scale]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
ans0 = tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ans1 = loss_fn(y_train[:1], predictions).numpy()
print(ans1)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

start = time.time()

model.fit(x_train, y_train, epochs=5)
# model.fit(x_train, y_train, epochs=1)

ans2 = loss_fn(y_train[:1], predictions).numpy()
ans = model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

ans = probability_model(x_test[:1])
duration = time.time()-start

print(ans0, ans1, ans2, ans)
print('duration:', duration)
