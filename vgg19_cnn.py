import tensorflow as tf
import numpy as np
import pandas as pd
import keras

dataname = 'fire'
col_names=['Image','Fire','Smoke']
data_dir = '../files/data/'



target_size_1 = (256, 256)
target_size_2 = (1,256,256,3)

df_train = pd.read_csv(data_dir + "fire/train/_train.csv", header=0, names=col_names)
df_test = pd.read_csv(data_dir + "fire/test/_test.csv", header=0, names=col_names)
df_val = pd.read_csv(data_dir + "fire/val/a_val.csv", header=0, names=col_names)




 # Train
x_train_temp = df_train["Image"]
x_train_aux = []

if dataname == 'fire':
    y_train_aux = df_train["Fire"]
else:
    y_train_aux = df_train["Smoke"]


for file_name in x_train_temp:
    img = tf.keras.preprocessing.image.load_img(data_dir + "fire/train/" + file_name, target_size =target_size_1)
    img = np.array(img)
    unit_ = img.reshape(target_size_2)
    x_train_aux.append(unit_)


x_train_aux = np.concatenate(x_train_aux,axis=0)
y_train_aux = np.reshape(np.array(y_train_aux),(np.array(y_train_aux).shape[0],1))

X_train = x_train_aux.astype('float32')/255.0
Y_train = np.array(list(list(zip(*y_train_aux))[0])) 
Y_train = np.reshape(Y_train, (len(Y_train), 1))
Y_train = keras.utils.to_categorical(Y_train, 2)

#Test 

x_test_temp = df_test["Image"]
x_test_aux = []

if dataname == 'fire':
    y_test_aux = df_test["Fire"]
else:
    y_test_aux = df_test["Smoke"]


for file_name in x_test_temp:
    img = tf.keras.preprocessing.image.load_img(data_dir + "fire/test/" + file_name, target_size =target_size_1)
    img = np.array(img)
    unit_ = img.reshape(target_size_2)
    x_test_aux.append(unit_)


x_test_aux = np.concatenate(x_test_aux,axis=0)
y_test_aux = np.reshape(np.array(y_test_aux),(np.array(y_test_aux).shape[0],1))

X_test = x_test_aux.astype('float32')/255.0
Y_test = np.array(list(list(zip(*y_test_aux))[0])) 
Y_test = np.reshape(Y_test, (len(Y_test), 1))
Y_test = keras.utils.to_categorical(Y_test, 2)



# Validation

x_val_temp = df_val["Image"]
x_val_aux = []

if dataname == 'fire':
    y_val_aux = df_val["Fire"]
else:
    y_val_aux = df_val["Smoke"]


for file_name in x_val_temp:
    img = tf.keras.preprocessing.image.load_img(data_dir + "fire/val/" + file_name, target_size =target_size_1)
    img = np.array(img)
    unit_ = img.reshape(target_size_2)
    x_val_aux.append(unit_)


x_val_test = x_val_aux
x_val_aux = np.concatenate(x_val_aux,axis=0)
y_val_aux = np.reshape(np.array(y_val_aux),(np.array(y_val_aux).shape[0],1))

X_val = x_val_aux.astype('float32')/255.0
Y_val = np.array(list(list(zip(*y_val_aux))[0])) 

Y_val = np.reshape(Y_val, (len(Y_val), 1))
Y_val = keras.utils.to_categorical(Y_val, 2)




model = tf.keras.applications.VGG19(include_top=False, input_shape=(256,256,3), pooling = 'avg')

predictions = keras.layers.Dense(2, activation='sigmoid')(model.output)
model = keras.models.Model(inputs=model.input, outputs=predictions)
model.summary()  #Line 2


model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  metrics=['accuracy'])

model.fit(x = X_train, y = Y_train, epochs=35, batch_size=64,validation_data=(X_val, Y_val))

loss3, acc3 = model.evaluate(X_train, Y_train, verbose=2)
print('Restored model, train accuracy: {:5.2f}%'.format(100 * acc3))

loss1, acc1 = model.evaluate(X_test, Y_test, verbose=2)
print('Restored model, test accuracy: {:5.2f}%'.format(100 * acc1))

loss2, acc2 = model.evaluate(X_val, Y_val, verbose=2)
print('Restored model, val accuracy: {:5.2f}%'.format(100 * acc2))


'''
            "cwd":"${workspaceFolder}/fire-smoke-AL/",

"cwd":"${workspaceFolder}/fire-smoke-AL/classification/",
            "args": ["--dataset", "fire", "--batch_size", "32"]
'''