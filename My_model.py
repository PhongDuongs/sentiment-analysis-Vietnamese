import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Flatten, LayerNormalization, GRU, Input, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import Data_preparation as dp

class MyModel:
    def __init__(self):
        self.dropout_threshold = 0.5
        self.input_dim = dp.data_prep.data_vocal_size
        self.output_dim = 64
        self.input_length = 800
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.model = None

    def build_model(self):
        input_layer = Input(shape=(self.input_length,))
        feature = Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length, embeddings_initializer=self.initializer)(input_layer)

        # Kiến trúc CNN
        cnn_feature = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(feature)
        cnn_feature = MaxPooling1D()(cnn_feature)
        cnn_feature = Dropout(self.dropout_threshold)(cnn_feature)
        cnn_feature = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(cnn_feature)
        cnn_feature = MaxPooling1D()(cnn_feature)
        cnn_feature = LayerNormalization()(cnn_feature)
        cnn_feature = Dropout(self.dropout_threshold)(cnn_feature)

        # Kiến trúc LSTM
        lstm_feature = Bidirectional(LSTM(32, dropout=self.dropout_threshold, return_sequences=True, kernel_initializer=self.initializer))(feature)
        lstm_feature = MaxPooling1D()(lstm_feature)

        lstm_feature = Bidirectional(GRU(32, dropout=self.dropout_threshold, return_sequences=True, kernel_initializer=self.initializer))(lstm_feature)
        lstm_feature = MaxPooling1D()(lstm_feature)
        lstm_feature = LayerNormalization()(lstm_feature)

        # Kết hợp các đặc trưng
        combined_feature = tf.keras.layers.concatenate([cnn_feature, lstm_feature])
        combined_feature = GlobalMaxPooling1D()(combined_feature)
        combined_feature = LayerNormalization()(combined_feature)

        # Thêm các lớp mạng nơ-ron
        classification_layer = Dense(256, activation='relu')(combined_feature)
        classification_layer = Dropout(0.2)(classification_layer)
        classification_layer = Dense(128, activation='relu')(classification_layer)
        classification_layer = Dropout(0.2)(classification_layer)
        classification_layer = Dense(64, activation='relu')(classification_layer)
        classification_layer = Dropout(0.2)(classification_layer)
        classification_layer = Dense(2, activation='softmax')(classification_layer)

        self.model = Model(inputs=input_layer, outputs=classification_layer)

    def compile_model(self, learning_rate=0.001):
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=20):
        checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1) # lưu mô hình tốt nhất
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1) # dừng sớm nếu không có cải thiện

        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping]) # huấn luyện mô hình

    def load_saved_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Khởi tạo mô hình
my_model = MyModel()
my_model.build_model()
my_model.compile_model(learning_rate=0.0001)  # Đề xuất: Giảm tốc độ học

# Huấn luyện mô hình
X_train = dp.data_prep.X_train
y_train = dp.data_prep.y_train
X_val = dp.data_prep.X_val
y_val = dp.data_prep.y_val

my_model.train_model(X_train, y_train, X_val, y_val, batch_size=64, epochs=30)  # Đề xuất: Tăng số lượng epochs, tăng kích thước batch

# Lưu mô hình
my_model.model.save('trained_model_part3.h5')

loaded_model = MyModel()
loaded_model.load_saved_model('trained_model_part3.h5')
