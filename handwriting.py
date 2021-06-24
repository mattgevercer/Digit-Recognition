import matplotlib.pyplot as plt
import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test)=mnist.load_data()

#normalize the pixel values in the image array data
x_train= tf.keras.utils.normalize(x_train, axis=1)
x_test= tf.keras.utils.normalize(x_test, axis=1)

#define the model: a flatten layer, 2 hidden layers with 128 neurons each, and an output layer
model= tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                   tf.keras.layers.Dense(units=128,activation=tf.nn.relu),
                                   tf.keras.layers.Dense(units=128,activation=tf.nn.relu),
                                   tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
loss,accuracy =model.evaluate(x_test, y_test)

#define CNN model and compare results
model1= tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3), activation='relu',input_shape = (28,28,1)),
                                   tf.keras.layers.MaxPooling2D(2,2),
                                   tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                   tf.keras.layers.MaxPooling2D(2,2),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(units=64,activation=tf.nn.relu),
                                   tf.keras.layers.Dense(units=64,activation=tf.nn.relu),
                                   tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)]
    )
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history=model1.fit(x_train, y_train, epochs=5,validation_data=(x_test,y_test))
loss1,accuracy1 =model1.evaluate(x_test, y_test)

print("Model0 accuracy is ",accuracy)
print("Model1 accuracy is ",accuracy1)
print("Model0 loss is ", loss)
print("Model1 loss is ",loss1)

#visualize how validation accuracy changes over the epochs
acc     = history.history[    'accuracy' ]
val_acc = history.history['val_accuracy' ]
epochs   = range(len(acc))
plt.plot  ( epochs, acc, label='Training Accuracy' )
plt.plot  ( epochs, val_acc, Label='Validation Accuracy' )
plt.title ('Training and Validation Accuracy Over Epochs')
plt.legend(loc="upper right")
plt.ylim(0.5, 1.5)
