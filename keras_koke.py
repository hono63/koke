from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from scipy.misc import imresize
import time
import os

THIS_FILE = os.path.dirname(__file__)

IMAGE_SIZE = 50
EPOCH_NUM  = 200

def create_cnn(folder):
    X = []
    Y = []
    folder = [f for f in folder if not f.startswith(".")]
    print("Target folders: " + str(folder))

    for index, name in enumerate(folder):
        dir = "./cnn/" + name
        files = glob.glob(dir + "/*.jpeg")
        for file in files:
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            data = np.asarray(image)
            X.append(data)
            Y.append(index)


    dense_size  = len(folder)
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype('float32')
    X = X / 255.0

    Y = np_utils.to_categorical(Y, dense_size)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size))
    model.add(Activation('softmax'))

    model.summary()

    return model, X_train, y_train

def learn_model(model, X_train, y_train, epochs):
    """Learn CNN model.
    * 0.01 sec / epoch @ Desktop PC with GPU. CUDA10.1
    * 1.1  sec / epoch @ Desktop PC without GPU.
    * 6    sec / epoch @ Macbook Pro.
    """
    optimizers ="Adadelta"
    model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
    results = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs )

    model_json_str = model.to_json()
    open('mnist_mlp_model.json', 'w').write(model_json_str)
    model.save_weights('mnist_mlp_weights.h5')

    return results

def load_model():
    model = None
    with open('mnist_mlp_model.json', 'r') as f:
        json_str = f.read()
        model = model_from_json(json_str)
        model.load_weights('mnist_mlp_weights.h5')
    return model

def show_graph(results, epochs):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = range(epochs)
    ax1.plot(x, results.history['accuracy'], label="Training Accuracy")
    ax1.plot(x, results.history['val_accuracy'], label="Validation Accuracy")
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

    #ax2 = fig.add_subplot(212)
    #for k, result in results.items():
        #ax2.plot(x, result.history['val_accuracy'], label=k)
    #ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

    plt.savefig('accuracy_and_val_accuracy.jpg', bbox_inches='tight')
    plt.show()
    plt.close()

def predict_img(img_name, model):
    img_arr = mpimg.imread(img_name)
    #resize_img = imresize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
    resize_img = np.array(Image.fromarray(img_arr).resize((IMAGE_SIZE, IMAGE_SIZE), resample=2))
    x = np.expand_dims(resize_img, axis=0)
    y_pred = model.predict(x)
    if y_pred[0][0] > 0.5:
        result = 'Haigoke ' + str(y_pred[0][0]*100)[0:3] + '%'
    else:
        result = 'Kamojigoke ' + str(y_pred[0][1]*100)[0:3] + '%'
    print(result)
    print(str(y_pred[0][0]) + ", " + str(y_pred[0][1]))
    # show
    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    plt.title(result)
    plt.show()

from keras.utils import plot_model
def show_model(model):
    plot_model(model, to_file='model.png')
    print("Saved model image: [model.png]")

if __name__ == "__main__":
    if 1:
        folder = os.listdir(os.path.join(THIS_FILE, "cnn"))
        model, X_train, y_train = create_cnn(folder)
        results = learn_model(model, X_train, y_train, EPOCH_NUM)
        show_graph(results, EPOCH_NUM)
    else:
        model = load_model()
    show_model(model)
    for imgfile in glob.glob(os.path.join(THIS_FILE, "predict", "*.jpg")):
        predict_img(imgfile, model)
    #predict_img("haigoke_sample.jpeg", model)
    #predict_img("kamojigoke_sample.jpeg", model)
    #predict_img("my_haigoke_trim.jpg", model)
    #predict_img("haigoke_on_tatami_trim.jpg", model)
    #predict_img("wet_haigoke.jpg", model)
    #predict_img("my_kamoji1.jpg", model)
    #predict_img("my_kamoji2.jpg", model)