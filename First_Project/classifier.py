from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, MaxPooling2D
import numpy as np
from Confusion_matrix import confu_matrix
from load_save import *
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam

X_train = load('X_train')
X_test = load('X_test')
Y_train = load('Y_train')
Y_test = load('Y_test')

def cnn(X_train,X_test,Y_train,Y_test):

    # reshaping data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid', input_shape=X_train[1].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2, batch_size=400, verbose=1)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    #y_predict_train = np.argmax(model.predict(X_train), axis=1)
    return y_predict,confu_matrix(Y_test, y_predict,0)#confu_matrix(Y_train, y_predict_train,1)

def ann(X_train,X_test,Y_train,Y_test):

    model = Sequential()
    model.add(Dense(20, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2, batch_size=100, verbose=1)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    #y_predict_train = np.argmax(model.predict(X_train), axis=1)
    return y_predict, confu_matrix(Y_test, y_predict,0), #confu_matrix(Y_train, y_predict_train,1)

def bi_lstm(X_train,X_test,Y_train,Y_test):

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))  # Three classes for sentiment: Positive, Negative, Neutral

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train,Y_train, epochs=1, batch_size=100,verbose=1)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    save('bi_lstm_y_pred',y_predict)
    print(y_predict)
    return y_predict

bi_lstm(X_train,X_test,Y_train,Y_test)


