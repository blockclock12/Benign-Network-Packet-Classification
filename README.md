# Benign-Network-Packet-Classification
for a given network packet, we verify the packet by an AI model(ET-BERT) and show the information about the packet how dangerous it is and other details  
codes are run from google-colab  

## Settings for application
```python  
!pip install streamlit pyngro  
!apt-get install -y tshark  
```

## Settings for model training
We used code from [ET-BERT](https://github.com/linwhitehat/ET-BERT)
the execution code be like...
```python
!python3 /content/drive/MyDrive/ET-BERT-main/fine-tuning/run_classifier.py --pretrained_model_path /content/drive/MyDrive/ET-BERT-main/models/pre-trained_model.bin \
                                   --output_model_path /content/drive/MyDrive/ET-BERT-main/models/output_model.bin \
                                   --vocab_path /content/drive/MyDrive/ET-BERT-main/models/encryptd_vocab.txt \
                                   --train_path /content/drive/MyDrive/ET-BERT-main/datasets/CSTNET-TLS1.3/packet/train_dataset.tsv \
                                   --dev_path /content/drive/MyDrive/ET-BERT-main/datasets/CSTNET-TLS1.3/packet/valid_dataset.tsv \
                                   --test_path /content/drive/MyDrive/ET-BERT-main/datasets/CSTNET-TLS1.3/packet/test_dataset.tsv \
                                   --config_path /content/drive/MyDrive/ET-BERT-main/models/bert/base_config.json \
                                   --epochs_num 10 --batch_size 500 --embedding word_pos_seg \
                                   --encoder transformer --mask fully_visible \
                                   --seq_length 128 --learning_rate 2e-5
```

## Code using order
1. train model by datasets and get model file
2. run app.py to run an application
3. run authorization_ngrok.py to run server and connect ngrok

## Datasets
We merged 3 different datasets of malicious and benign packet
1. CICIDS2018
2. NSL-KDD
3. CTU-13


## files
[ET-BERT link](https://github.com/linwhitehat/ET-BERT)
[you can get trained-model file here](https://drive.google.com/file/d/1sf1gK-DmWTOscYO2ejTtmK30Z3mpCigb/view)
[app running colab environment](https://colab.research.google.com/drive/16azfijc3aZQONIrOUFA8y0itaCXqnSLS?usp=sharing#scrollTo=C0Kq5RGzmgIX)
[dataset merge colab environment](https://colab.research.google.com/drive/1svKCuhvnzxU_1nDqZ1ST8ZWZb_gmsB7U?usp=sharing)
