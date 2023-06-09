import VietnameseTextPreprocessor as Vi
import Data_preparation as dp
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
Vitoken = Vi.VietnameseTextProcessor(abbreviations_file='abbreviations.csv')

def prediction(input_text, tokenizer, model):
    input_model = Vitoken.process_text(input_text)
    print('text processed:', input_model)
    tokenized_data = tokenizer.texts_to_sequences([input_model])
    vec_data = pad_sequences(tokenized_data, padding='post', maxlen=800)
    result, conf = inference_model(vec_data, model)
    return result, conf


def inference_model(input, model):
    output = model(input).numpy()[0]
    result = output.argmax()
    conf = float(output.max())
    label_dict = {0: 'tích cực', 1: 'tiêu cực'}
    label = list(label_dict.values())
    return label[result], conf

my_model = tf.keras.models.load_model('trained_model_part3.h5')
with open('tokenizer_data.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
'''
print(prediction('tôi dẫn gia đình đi ăn quán này và nhân viên phục vụ rất tốt', tokenizer, my_model))
print(prediction('trận thể thao này thật sự rất đang xấu hổ, đừng rủ tôi xem đội này nữa', tokenizer, my_model))
print(prediction('món này rất ngon nhaaa', tokenizer, my_model))
print(prediction('món này không ổn cho lắm', tokenizer, my_model))
'''
while True:
    input_text = input('Nhập câu cần dự đoán: ')
    if input_text == 'exit':
        break
    print(prediction(input_text, tokenizer, my_model))