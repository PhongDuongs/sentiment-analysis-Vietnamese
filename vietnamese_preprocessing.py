import re
import pandas as pd
from underthesea import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def pair_vietnamese_words(text):
    processed_text = word_tokenize(text, format="text") 
    return processed_text

def remove_stopwords(text):
    stopwords = ["là", "của", "được", "và", "với", "một", "các", "cho", "này", "làm", "để", "trên", "như", "đã", "nơi", "thì", "đến", "rằng", "nên", "về", "cũng", "nhưng", "cũng", "nếu", "đó", "từ", "ở", "hay", "vì", "đấy", "khi", "theo", "lại", "nào", "đều", "để", "vậy", "được", "vừa", "vẫn", "sẽ", "cứ", "cho", "vì", "điều", "mà", "thì", "lúc", "từ", "khi", "của", "chỉ", "rồi", "đang"]

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def normalize_abbreviations(text, abbreviations):
    pattern = r'(\w+)(\W+)'  # Mẫu tìm kiếm từ đứng trước ký tự đặc biệt
    normalized_text = re.sub(pattern, lambda m: abbreviations.get(m.group(1).lower(), m.group(1)) + m.group(2), text)
    return normalized_text

def remove_single_characters(text):
    words = text.split()
    processed_words = []
    for word in words:
        if len(word) == 1 and not word.isdigit():
            continue    
        processed_words.append(word)
    processed_text = ' '.join(processed_words)
    return processed_text


def handle_repeated_syllables(text):
    words = text.split()
    processed_words = []
    
    for word in words:
        processed_word = ""
        current_char = ""
        for char in word:
            if char.isalpha():  # Chỉ xử lý các ký tự là chữ cái
                if char != current_char: # Nếu ký tự hiện tại khác ký tự trước đó thì thêm vào chuỗi kết quả
                    processed_word += char 
                current_char = char
            else:
                processed_word += char  # Giữ nguyên các ký tự không phải chữ cái
        
        processed_words.append(processed_word)
    
    processed_text = ' '.join(processed_words)
    return processed_text

def remove_diacritics(text):
    # Tổng cộng có 67 trường hợp dấu trong tiếng Việt
    diacritic_chars = "áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    # Sẽ được thay thế bằng từng ký tự tương ứng trong chuỗi này
    non_diacritic_chars = "aaaaaaaaaaaaaaaaadeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyy"
    for i in range(len(diacritic_chars)):
        text = text.replace(diacritic_chars[i], non_diacritic_chars[i])
    return text

def lower_text(text):
    lower_text = text.lower()
    return lower_text

def remove_special_characters_vietnamese(text):
    # Loại bỏ các ký tự đặc biệt trong tiếng Việt
    text = re.sub(r"[^a-zA-Z0-9\sáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]+", "", text)
    return text

def remove_specific_words(text):
    # Loại bỏ các từ không có trong bảng chữ cái tiếng Việt
    words_to_remove = ["f", "z", "j", "w"]
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in words_to_remove]
    filtered_text = " ".join(filtered_words)
    return filtered_text

# Ví dụ sử dụng
# Đường dẫn đến file CSV chứa từ viết tắt và từ tương ứng
abbreviations_file = 'abbreviations.csv'

# Đọc dữ liệu từ file CSV
abbreviations_df = pd.read_csv(abbreviations_file)

# Chuyển đổi dữ liệu thành từ điển
abbreviations = dict(zip(abbreviations_df['Abbreviation'], abbreviations_df['Expansion']))
sentence = pd.read_csv('data.csv')
print(sentence["sentence"][0])
data_text = []
for i in range(len(sentence["sentence"])):
    sentence["sentence"][i] = lower_text(sentence["sentence"][i]) #bước 1 chuyển thành chữ thường
    sentence["sentence"][i] = handle_repeated_syllables(sentence["sentence"][i]) #bước 2 xử lý âm tiết lặp
    sentence["sentence"][i] = remove_special_characters_vietnamese(sentence["sentence"][i]) #bước 3 xóa ký tự đặc biệt
    sentence["sentence"][i] = normalize_abbreviations(sentence["sentence"][i], abbreviations) #bước 3 chuẩn hóa từ viết tắt
    sentence["sentence"][i] = remove_single_characters(sentence["sentence"][i]) #bước 4 xóa dấu
    #sentence["sentence"][i] = remove_stopwords(sentence["sentence"][i]) #bước 5 xóa stopword
    sentence["sentence"][i] = pair_vietnamese_words(sentence["sentence"][i]) #bước 6 xóa ký tự đơn
    #sentence["sentence"][i] = remove_diacritics(sentence["sentence"][i]) #bước 7 xóa dấu tiếng việt
    data_text.append(sentence["sentence"][i])
    print(sentence["sentence"][i])

#print(data_text)

tokenizer_data = Tokenizer(oov_token="<OOV>", filters = '',split = ' ')
tokenizer_data.fit_on_texts(data_text)
tokenizer_data_text = tokenizer_data.texts_to_sequences(data_text)
vec_data = pad_sequences(tokenizer_data_text, padding='post', maxlen=20)
data_vocal_size = len(tokenizer_data.word_index) + 1
print(vec_data)
print(vec_data.shape)
print(tokenizer_data.word_index)