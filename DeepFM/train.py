# import sys

# sys.path.append("../")
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy

from model import DeepFM
from criteo import create_criteo_dataset

import os

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = '../dataset/criteo_sampled_data.csv'
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================ Build Model==========================
    model = DeepFM(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    model.summary()
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[BinaryAccuracy()])
    # ============================model checkpoint======================
    # logdir = './logs'
    # tensorboard = TensorBoard(logdir)
    check_path = './model_output/deepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint_dir = os.path.dirname(check_path)
    checkpoint = ModelCheckpoint(check_path,
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 verbose=1,
                                 save_best_only=True,
                                 period=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    # 保存训练过程中所有标量指标
    csv_logger = tf.keras.callbacks.CSVLogger(filename='./logs/train_log.csv', separator=',')
    # ==============================Fit==============================
    model.fit(train_X,
              train_y,
              epochs=epochs,
              verbose=1,
              callbacks=[earlystop, checkpoint, csv_logger],
              batch_size=batch_size,
              validation_split=0.1)
    # ===========================Test==============================
    print('test acc: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    best_model = DeepFM(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    best_model.compile(loss=binary_crossentropy,
                       optimizer=Adam(learning_rate=learning_rate),
                       metrics=[BinaryAccuracy()])
    print(latest)
    best_model.load_weights(latest)
    print('test acc: %f' % best_model.evaluate(test_X, test_y, batch_size=batch_size)[1])
