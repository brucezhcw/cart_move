import tensorflow as tf
from _readertools import batch_generator
from _model import DrawLSTM
import os
import win_unicode_console
win_unicode_console.enable()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_pics', 1, 'number of seqs in one batch')
tf.flags.DEFINE_integer('widths_size', 50, 'number of seqs in one batch')
tf.flags.DEFINE_integer('timestep', 26, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 361, 'size of embedding')
tf.flags.DEFINE_float('train_keep_prob', 1.0, 'dropout rate during training')
tf.flags.DEFINE_integer('num_classes', 361, 'max char number')

tf.flags.DEFINE_string('checkpoint_path', 'C:\\Users\\brucezhcw\\Desktop\\processed\\model\\', 'checkpoint path')

def main(_):
    model_path = os.path.join(FLAGS.checkpoint_path, '-21500')
    picture_path = 'C:\\Users\\brucezhcw\\Desktop\\processed\\cart\\'
    label_path = 'C:\\Users\\brucezhcw\\Desktop\\processed\\label\\'
    if os.path.exists(FLAGS.checkpoint_path) is False:
        print('please specify the dir of the saved model !!!')
        return

    g = batch_generator(picture_path, label_path, FLAGS.num_pics)

    model = DrawLSTM(is_train=False,
                    num_classes=FLAGS.num_classes,
                    num_pics=FLAGS.num_pics,
                    widths_size=FLAGS.widths_size,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size,
                    timestep=FLAGS.timestep
                    )

    model.load(model_path)

    model.test(g)

if __name__ == '__main__':
    tf.app.run()
