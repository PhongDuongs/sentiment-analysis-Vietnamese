from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Flatten, LayerNormalization, GRU, Input, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
import Data_preparation as dp


def model():
    dropout_threshold = 0.5
    input_dim = dp.data_prep.data_vocal_size
    output_dim = 32
    input_length = 3400
    initializer = tf.keras.initializers.GlorotNormal(seed=123)

    input_layer = Input(shape=(input_length))
    feature = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length, embeddings_initializer=initializer)(input_layer)

    
    cnn_feature = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(feature)
    cnn_feature = MaxPooling1D()(cnn_feature)
    cnn_feature = Dropout(dropout_threshold)(cnn_feature)
    cnn_feature = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(cnn_feature)
    cnn_feature = MaxPooling1D()(cnn_feature)
    cnn_feature = LayerNormalization()(cnn_feature)
    cnn_feature = Dropout(dropout_threshold)(cnn_feature)

    lstm_feature = Bidirectional(LSTM(64, dropout = dropout_threshold, return_sequences = True, kernel_initializer = initializer))(feature)
    lstm_feature = MaxPooling1D()(lstm_feature)

    lstm_feature = Bidirectional(GRU(64, dropout = dropout_threshold, return_sequences = True, kernel_initializer = initializer))(lstm_feature)
    lstm_feature = MaxPooling1D()(lstm_feature)
    lstm_feature = LayerNormalization()(lstm_feature)

    combined_feature = tf.keras.layers.concatenate([cnn_feature, lstm_feature])
    combined_feature = GlobalMaxPooling1D()(combined_feature)
    combined_feature = LayerNormalization()(combined_feature)

    classification_layer = Dense(128, activation='relu')(combined_feature)
    classification_layer = Dropout(0.2)(classification_layer)
    classification_layer = Dense(90, activation='relu')(classification_layer)
    classification_layer = Dropout(0.2)(classification_layer)
    classification_layer = Dense(70, activation='relu')(classification_layer)
    classification_layer = Dropout(0.2)(classification_layer)
    classification_layer = Dense(50, activation='relu')(classification_layer)
    classification_layer = Dropout(0.2)(classification_layer)
    classification_layer = Dense(30, activation='relu')(classification_layer)
    classification_layer = Dropout(0.2)(classification_layer)
    classification_layer = Dense(15, activation='relu')(classification_layer)
    classification_layer = Dropout(0.2)(classification_layer)
    classification_layer = Dense(2, activation='softmax')(classification_layer)
    
    model = tf.keras.Model(inputs = input_layer, outputs = classification_layer)

    return model

model = model()
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())

