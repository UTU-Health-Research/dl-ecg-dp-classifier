# Secure ECG classification using deep learning with diffrential privacy

Steps: 
  - Download the ECG and do preprocessing by following the steps in https://github.com/UTU-Health-Research/dl-ecg-classifier and save the processed data in the data folder
  - run the training script ``` dp_with_all_hospital_data.py ``` to train the model with differential privacy for the ResNet-based architectures.

Note: the training predictions & logs are saved on weight&bias platform, so you have to create an account https://wandb.ai/ and connect it with your code

# Code implemenation for the Transformer model can be found at this githup repo 
  https://github.com/UTU-Health-Research/dl-ecg-classifier/tree/transformer_network

to implement the transformer with differential privacy we created a new files:
  - src/modeling/models/ctn_dp.py 
  - src/modeling/train_utils_dp.py
  - src/modeling/predict_utils.py

  to train the transformer run the script ctn_dp_train.py and for evaluting the model on the test set run this script ctn_predict.py
