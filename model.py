from Read_file import *
from Prepocessing import *


def apply_model():
    df = final_table()
    X = df[0]
    Y = df[1]

    maxlen = 22
    T = 4

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    model_lstm = Sequential()
    model_lstm.add(LSTM(256, input_shape=(T, maxlen)))
    model_lstm.add(Dense(256, activation='sigmoid'))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Dense(maxlen, activation='softmax'))

    model_lstm.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    history = model_lstm.fit(X_train, y_train, epochs=100, validation_split=0.1, batch_size=64)

    plt.clf()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'y', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    p_train = model_lstm.predict(X_train)
    p_test = model_lstm.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(p_train, axis=1))
    test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(p_test, axis=1))
    print(train_acc)
    print(test_acc)
