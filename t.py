import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf


training_epochs = 5000
n_dim = 193
n_classes = 10
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

init = tf.global_variables_initializer()


class NN:
    def __init__(self):
        self.filename = os.path.abspath(os.path.join(os.sep, 'tmp', 'model.ckpt'))
        self.saver = tf.train.Saver()
        self.session = None

    def train(self, x, y):
        print("Start processing")
        features, labels = self.process_data(x, y)
        print("Start splitting")
        train_x, train_y, test_x, test_y = self.split_train_set(features, labels)

        cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        cost_history = np.empty(shape=[1], dtype=float)
        self.session = tf.Session()
        self.session.run(init)
        print("Start training")
        for epoch in range(training_epochs):
            _, cost = self.session.run([optimizer, cost_function], feed_dict={X: train_x, Y: train_y})
            cost_history = np.append(cost_history, cost)
        print('Test accuracy: ', round(self.session.run(accuracy, feed_dict={X: test_x, Y: test_y}), 3))

    def predict(self, x):
        return self.session.run(tf.argmax(y_, 1), feed_dict={X: x})

    def load(self):
        self.session = tf.Session()
        self.saver.restore(self.session, self.filename)

    def save(self):
        self.saver.save(self.session, self.filename)

    def split_train_set(self, features, labels):
        labels = self.one_hot_encode(labels)
        train_test_split = np.random.rand(len(features)) < 0.70
        train_x = features[train_test_split]
        train_y = labels[train_test_split]
        test_x = features[~train_test_split]
        test_y = labels[~train_test_split]
        return train_x, train_y, test_x, test_y

    @staticmethod
    def extract_features(y, sr):
        stft = np.abs(librosa.stft(y))
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        return mfccs, chroma, mel, contrast, tonnetz

    def process_data(self, x, labels):
        m = np.empty((0, n_dim))
        _labels = np.empty(0)
        for data, label in zip(x, labels):
            y, sr = data
            try:
                features = self.extract_features(y, sr)
            except Exception:
                continue
            m = np.vstack([m, np.hstack(features)])
            _labels = np.append(_labels, label)
        return m, np.array(_labels, dtype=np.int)

    @staticmethod
    def one_hot_encode(labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode


def main():
    sounds = []
    labels = []
    sound_files = glob.glob(os.path.join('sounds', 'car_horn', '*.wav'))

    for i, filename in enumerate(sound_files):
        sounds.append(librosa.load(filename))
        labels.append(filename.split('/')[2].split('-')[1])
        if i % 100 == 0:
            print("{}, {} left".format(filename, len(sound_files) - i))

    nn = NN()
    nn.train(sounds, labels)

    features = np.empty((0, n_dim))
    y, sr = librosa.load(os.path.join('sounds', 'car_horn', '7061-6-0-0.wav'))
    ext_features = np.hstack(NN.extract_features(y, sr))
    mytest_x = np.array(np.vstack([features, ext_features]))

    print(nn.predict(mytest_x))


if __name__ == '__main__':
    import glob

    main()
