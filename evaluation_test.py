from tensorflow.keras.models import load_model
import Data_preparation as dp
import numpy as np
# Import model đã huấn luyện từ file trained_model.h5
loaded_model = load_model('trained_model_part3.h5')

# Đánh giá mô hình trên tập dữ liệu X_test và y_test
evaluation = loaded_model.evaluate(dp.data_prep.X_test, dp.data_prep.y_test)

# In kết quả đánh giá
print('Loss trên tập test:', evaluation[0])
print('Accuracy trên toàn bộ tập test:', evaluation[1])


# Dự đoán nhãn cho tập dữ liệu X_test

y_pred = loaded_model.predict(dp.data_prep.X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Chuyển đổi nhãn dự đoán từ số sang văn bản
label_id_to_text = {v: k for k, v in dp.data_prep.label_to_id.items()}
y_pred_text = [label_id_to_text[label_id] for label_id in y_pred_labels]

# Tính accuracy cho từng nhãn
label_0_indices = np.where(dp.data_prep.y_test[:, 0] == 1)[0]
label_1_indices = np.where(dp.data_prep.y_test[:, 1] == 1)[0]

accuracy_label_0 = np.mean(np.array(y_pred_text)[label_0_indices] == 'tích cực')
accuracy_label_1 = np.mean(np.array(y_pred_text)[label_1_indices] == 'tiêu cực')


# In kết quả accuracy
print('Accuracy for label Tích cực:', accuracy_label_0)
print('Accuracy for label Tiêu cực:', accuracy_label_1)

