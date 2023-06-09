import os
import csv
import codecs
import pandas as pd
# Đường dẫn đến thư mục chứa các file .txt
folder_path = '/sentiment/data_train/data_train/train/pos'

# Đường dẫn đến file csv để ghi nội dung
csv_file_path = 'data_train.csv'

# Mở file csv để ghi
csv_file = codecs.open(csv_file_path, 'w', 'utf-8-sig')  # Sử dụng codecs.open với mã hóa utf-8-sig
csv_writer = csv.writer(csv_file)

# Ghi dòng tiêu đề vào file csv
csv_writer.writerow(['sentence', 'label'])

# Lặp qua các file trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # Mở file .txt để đọc nội dung
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            content = txt_file.read()

            # Ghi nội dung của file .txt vào file csv với nhãn 'tích cực'
            csv_writer.writerow([content.strip(), 'tích cực'])

# Đường dẫn đến thư mục chứa các file .txt tiêu cực
folder_path_neg = '/sentiment/data_train/data_train/train/neg'

# Lặp qua các file trong thư mục tiêu cực
for filename in os.listdir(folder_path_neg):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path_neg, filename)

        # Mở file .txt để đọc nội dung
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            content = txt_file.read()

            # Ghi nội dung của file .txt vào file csv với nhãn 'tiêu cực'
            csv_writer.writerow([content.strip(), 'tiêu cực'])

# Đóng file csv
csv_file.close()


# Đường dẫn đến file CSV gốc
csv_file_path = 'data_train.csv'

# Đọc file CSV vào DataFrame với encoding 'utf-8-sig'
df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

# Thay thế ký tự "_" bằng khoảng trắng trong cột "content"
df['sentence'] = df['sentence'].str.replace('_', ' ')

# Đường dẫn đến file CSV đã chỉnh sửa
output_csv_file_path = 'data_train_modified.csv'

# Ghi DataFrame đã chỉnh sửa vào file CSV với encoding 'utf-8-sig'
df.to_csv(output_csv_file_path, index=False, encoding='utf-8-sig')
