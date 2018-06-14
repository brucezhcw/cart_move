import tensorflow as tf
from _readertools import batch_generator
from _model import DrawLSTM
import os
import codecs
import win_unicode_console
win_unicode_console.enable()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_pics', 64, 'number of seqs in one batch')
tf.flags.DEFINE_integer('widths_size', 50, 'number of seqs in one batch')
tf.flags.DEFINE_integer('timestep', 26, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 361, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_integer('grad_clip', 5, 'size of embedding')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_integer('max_steps', 50000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 500, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 50, 'log to the screen every n steps')
tf.flags.DEFINE_integer('num_classes', 361, 'max char number')

def main(_):
    model_path = 'C:\\Users\\brucezhcw\\Desktop\\processed\\model\\'
    picture_path = 'C:\\Users\\brucezhcw\\Desktop\\processed\\cart\\'
    label_path = 'C:\\Users\\brucezhcw\\Desktop\\processed\\label\\'
    if os.path.exists(model_path) is False:
        print('making dir: %s' % model_path)
        os.makedirs(model_path)

    g = batch_generator(picture_path, label_path, FLAGS.num_pics)

    model = DrawLSTM(is_train=True,
                    num_classes=FLAGS.num_classes,
                    num_pics=FLAGS.num_pics,
                    widths_size=FLAGS.widths_size,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    grad_clip=FLAGS.grad_clip,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size,
                    timestep=FLAGS.timestep
                    )
    model.train(batch_generator=g,
                max_steps=FLAGS.max_steps,
                save_path=model_path,
                save_every_n=FLAGS.save_every_n,
                log_every_n=FLAGS.log_every_n,
                )

if __name__ == '__main__':
    tf.app.run()
