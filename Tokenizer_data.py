import VietnameseTextPreprocessor as Vi
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split


# Chuỗi văn bản đầu vào
data_sentiment = pd.read_csv('data.csv')

input_text = data_sentiment['sentence'].values
input_label = data_sentiment['label'].values

label_to_id = {'tích cực': 0, 'tiêu cực': 1}
# Ví dụ sử dụng
abbreviations_file = 'abbreviations.csv'
processor = Vi.VietnameseTextProcessor(abbreviations_file)

input_pre = []
label_pre = []
# Chuỗi văn bản đầu vào
for idx, text in enumerate(input_text):
    input_ = processor.process_text(text)
    input_pre.append(input_)
    label_pre.append(input_label[idx])

label_idx = [label_to_id[label] for label in label_pre]
label_tf = tf.keras.utils.to_categorical(label_idx, num_classes=2, dtype='float32')
tokenizer_data = Tokenizer(oov_token="<OOV>", filters = '',split = ' ')
tokenizer_data.fit_on_texts(input_pre)
tokenizer_data_text = tokenizer_data.texts_to_sequences(input_pre)
vec_data = pad_sequences(tokenizer_data_text, padding='post', maxlen=15)
pickle.dump(tokenizer_data, open('tokenizer_data.pkl', 'wb'))
data_vocal_size = len(tokenizer_data.word_index) + 1

X_train, X_val, y_train, y_val = train_test_split(vec_data, label_tf, test_size=0.3, stratify=label_tf, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train, random_state=101)
print(vec_data.shape)
print(data_vocal_size)
print(label_tf[0:10])
print(X_train)
print(y_train)
print(X_test)
print(y_test)
print(X_val)
print(y_val)