import re
import pandas as pd
from underthesea import word_tokenize

class VietnameseTextProcessor:
    def __init__(self, abbreviations_file):
        self.abbreviations = self.load_abbreviations(abbreviations_file)

    def load_abbreviations(self, abbreviations_file):
        abbreviations_df = pd.read_csv(abbreviations_file)
        abbreviations = dict(zip(abbreviations_df['Abbreviation'], abbreviations_df['Expansion']))
        return abbreviations

    def pair_vietnamese_words(self, text):
        processed_text = word_tokenize(text, format="text")
        return processed_text

    def remove_stopwords(self, text):
        stopwords = ["là", "của", "được", "và", "với", "một", "các", "cho", "này", "làm", "để", "trên", "như", "đã", "nơi",
                     "thì", "đến", "rằng", "nên", "về", "cũng", "nhưng", "cũng", "nếu", "đó", "từ", "ở", "hay", "vì", "đấy",
                     "khi", "theo", "lại", "nào", "đều", "để", "vậy", "được", "vừa", "vẫn", "sẽ", "cứ", "cho", "vì", "điều",
                     "mà", "thì", "lúc", "từ", "khi", "của", "chỉ", "rồi", "đang"]

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    def normalize_abbreviations(self, text):
        pattern = r'(\w+)(\W+)'  # Mẫu tìm kiếm từ đứng trước ký tự đặc biệt
        normalized_text = re.sub(pattern, lambda m: self.abbreviations.get(m.group(1).lower(), m.group(1)) + m.group(2),
                                 text)
        return normalized_text

    def remove_single_characters(self, text):
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) == 1 and not word.isdigit():
                continue
            processed_words.append(word)
        processed_text = ' '.join(processed_words)
        return processed_text

    def handle_repeated_syllables(self, text):
        words = text.split()
        processed_words = []

        for word in words:
            processed_word = ""
            current_char = ""
            for char in word:
                if char.isalpha():  # Chỉ xử lý các ký tự là chữ cái
                    if char != current_char:  # Nếu ký tự hiện tại khác ký tự trước đó thì thêm vào chuỗi kết quả
                        processed_word += char
                    current_char = char
                else:
                    processed_word += char  # Giữ nguyên các ký tự không phải chữ cái

            processed_words.append(processed_word)

        processed_text = ' '.join(processed_words)
        return processed_text

    def remove_diacritics(self, text):
        text = str(text)
        # Tổng cộng có 67 trường hợp dấu trong tiếng Việt
        diacritic_chars = "áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
        # Sẽ được thay thế bằng từng ký tự tương ứng trong chuỗi này
        non_diacritic_chars = "aaaaaaaaaaaaaaaaadeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyy"
        for i in range(len(diacritic_chars)):
            text = text.replace(diacritic_chars[i], non_diacritic_chars[i])
        return text

    def lower_text(self, text):
        text = str(text)
        lower_text = text.lower()
        return lower_text

    def remove_special_characters_vietnamese(self, text):
        text = str(text)
        # Loại bỏ các ký tự đặc biệt trong tiếng Việt
        text = re.sub(r"[^a-zA-Z0-9\sáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]+", "", text)
        return text

    def remove_specific_words(self, text):
        text = str(text)
        # Loại bỏ các từ không có trong bảng chữ cái tiếng Việt
        words_to_remove = ["f", "z", "j", "w"]
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in words_to_remove]
        filtered_text = " ".join(filtered_words)
        return filtered_text

    def process_text(self, text):
        text = str(text)
        text = self.lower_text(text)
        text = self.handle_repeated_syllables(text)
        text = self.remove_special_characters_vietnamese(text)
        text = self.normalize_abbreviations(text)
        text = self.remove_single_characters(text)
        #text = self.remove_stopwords(text)
        text = self.pair_vietnamese_words(text)
        #text = self.remove_diacritics(text)
        text = self.remove_specific_words(text)
        return text


