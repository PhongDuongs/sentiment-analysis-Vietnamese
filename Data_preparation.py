import VietnameseTextPreprocessor as Vi
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, input_text, input_label, label_to_id, abbreviations_file):
        self.input_text = input_text
        self.input_label = input_label
        self.label_to_id = label_to_id
        self.abbreviations_file = abbreviations_file
        self.processor = None
        self.tokenizer_data = None
        self.vec_data = None
        self.data_vocal_size = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

    def preprocess_text(self):
        # Đọc danh sách từ viết tắt
        abbreviations = pd.read_csv(self.abbreviations_file)

        # Tiền xử lý văn bản
        self.processor = Vi.VietnameseTextProcessor(abbreviations_file=self.abbreviations_file)
        self.input_pre = [self.processor.process_text(text) for text in self.input_text]

        # Chuyển đổi nhãn sang dạng số
        self.label_idx = [self.label_to_id[label] for label in self.input_label if label in self.label_to_id]
        self.label_tf = tf.keras.utils.to_categorical(self.label_idx, num_classes=2, dtype='float32')

    def tokenize_text(self):
        # Tạo tokenizer và vector hóa văn bản
        self.tokenizer_data = Tokenizer(oov_token="<OOV>", filters='', split=' ')
        self.tokenizer_data.fit_on_texts(self.input_pre)
        self.tokenizer_data_text = self.tokenizer_data.texts_to_sequences(self.input_pre)
        self.vec_data = pad_sequences(self.tokenizer_data_text, padding='post', maxlen=800)

        # Lưu kích thước từ vựng
        self.data_vocal_size = len(self.tokenizer_data.word_index) + 1

    def split_data(self, test_size=0.2, stratify=None, random_state=20):
        # Phân chia dữ liệu thành tập huấn luyện, tập kiểm tra và tập validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.vec_data, self.label_tf, test_size=test_size, stratify=stratify, random_state=random_state
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train, self.y_train, test_size=test_size, stratify=self.y_train, random_state=random_state
        )

        
    def save_tokenizer(self, tokenizer_file):
        # Lưu tokenizer vào tệp pickle
        with open(tokenizer_file, 'wb') as file:
            pickle.dump(self.tokenizer_data, file)


# Sử dụng lớp DataPreparation
data_sentiment = pd.read_csv('data_train_modified.csv')
input_text = data_sentiment['sentence'].values
input_label = data_sentiment['label'].values
label_to_id = {'tích cực': 0, 'tiêu cực': 1}
abbreviations_file = 'abbreviations.csv'

data_prep = DataPreparation(input_text, input_label, label_to_id, abbreviations_file)
data_prep.preprocess_text()
data_prep.tokenize_text()
data_prep.split_data(test_size=0.2, stratify=data_prep.label_tf, random_state=42)
data_prep.save_tokenizer('tokenizer_data.pkl')


# Truy cập dữ liệu đã được tạo
X_train = data_prep.X_train
X_val = data_prep.X_val
X_test = data_prep.X_test
y_train = data_prep.y_train
y_val = data_prep.y_val
y_test = data_prep.y_test
data_vocal_size = data_prep.data_vocal_size
vec_data = data_prep.vec_data
