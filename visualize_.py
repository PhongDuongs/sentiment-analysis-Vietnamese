import pandas as pd
import VietnameseTextPreprocessor as Vi
import matplotlib.pyplot as plt
data_sentiment = pd.read_csv('data_train_modified.csv')
#print(data_sentiment.head())

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
# Xử lý văn bản

'''

seq_len = [len(i.split()) for i in input_pre[0:1000]]
pd.Series(seq_len).hist(bins = 10)
plt.title('0:50')


plt.show()

seq_len = [len(i.split()) for i in input_pre[1000:2000]]
pd.Series(seq_len).hist(bins = 10)
plt.title('50:150')

plt.show()

seq_len = [len(i.split()) for i in input_pre[2000:3000]]
pd.Series(seq_len).hist(bins = 10)
plt.title('150:203')

plt.show()

seq_len = [len(i.split()) for i in input_pre[3000:5000]]
pd.Series(seq_len).hist(bins = 10)
plt.title('15000:20000')

plt.show()

seq_len = [len(i.split()) for i in input_pre[5000:6000]]
pd.Series(seq_len).hist(bins = 10)
plt.title('20000:25000')

plt.show()

seq_len = [len(i.split()) for i in input_pre[6000:7500]]
pd.Series(seq_len).hist(bins = 10)
plt.title('25000:30000')
plt.show()


'''

data_sentiment['label'].value_counts().plot(kind='bar')
plt.show()