from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense
from keras.models import model_from_json
import numpy as np

data_dim = 16
timesteps = 8
nb_classes = 10

encoder_a = Sequential()
encoder_a.add(LSTM(32, input_shape=(timesteps, data_dim)))

encoder_b = Sequential()
encoder_b.add(LSTM(32, input_shape=(timesteps, data_dim)))

decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
decoder.add(Dense(32, activation='relu'))
decoder.add(Dense(nb_classes, activation='softmax'))

decoder.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

print decoder.layers

# generate dummy training data
x_train_a = np.random.random((1000, timesteps, data_dim))
x_train_b = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, nb_classes))

# generate dummy validation data
x_val_a = np.random.random((100, timesteps, data_dim))
x_val_b = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, nb_classes))


decoder.fit([x_train_a, x_train_b], y_train,
            batch_size=64, nb_epoch=1,
            validation_data=([x_val_a, x_val_b], y_val))

# later...
print "saving model..."
json_file = open('model.json', 'w')
json_file.write(decoder.to_json())
json_file.close()

print decoder.layers

print "saving weights..."
decoder.save_weights('model.h5')

print decoder.layers

# load json and create model
print "loading model from json..."
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

print loaded_model.layers

# load weights into new model
print "loading weights from h5..."
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

print loaded_model.layers

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
