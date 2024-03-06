import tensorflow as tf
from config import *
from vocabulary import Vocabulary
from vectorizer import Vectorizer
from model import AttentionOCR
from data_generator import CSVDataSource
import functools
if __name__ == '__main__':
    max_txt_length = 30
    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=320, max_txt_length=max_txt_length)
    model = AttentionOCR(vocabulary=voc, max_txt_length=max_txt_length)

    train_data = CSVDataSource(vec, 'data/', 'train.txt', is_training=True)
    validation_data = CSVDataSource(vec, 'data/', 'validation.txt')
    # for x in functools.partial(CSVDataSource().method, 10)():
    #     print(x)

    train_gen = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32, tf.float32))
    validation_gen = tf.data.Dataset.from_generator(validation_data, output_types=(tf.float32, tf.float32, tf.float32))
    model.fit_generator(train_gen, epochs=3, batch_size=1, validation_data=validation_gen, validate_every_steps=10)
    model.save('model.h5')

    # for image, decoder_input, decoder_output in validation_data():
    #     txt = voc.one_hot_decode(decoder_output, max_txt_length)
    #     pred = model.predict([image])[0]
    #     model.visualise([image])
    #     print(txt, "prediction: ", pred)