
# Sequential Emotion Recognition in Conversations

FNU Hardeep, Harish Babu Manogaran, Mridul Khurana, Shri Sarvesh Venkatachala Moorthy and Upasana Sivaramakrishnan

Fall 2022 CS 5824/ECE 5424 Advanced Machine Learning:  
Course Project - Virginia Tech

### Required Packages:
------
transformers=4.14.1

torch=1.8

vocab=0.0.5

numpy

tqdm

sklearn

pickle

pandas

### Datasets:
------
The Following datasets are available at `data/`
```
data/
└───meld/
│   │   dev_sent_emo.csv
│   │   test_sent_emo.csv
|   |   train_sent_emo.csv
│   
└───emorynlp/
│   │   emorynlp_dev_final.csv
│   │   emorynlp_test_final.csv
|   |   emorynlp_train_final.csv
│   
└───iemocap/
│   │   iemocap_dev.txt
│   │   iemocap_test.txt
|   |   iemocap_train.txt
|
└───daily_dialog/
│   │   dailydialog_dev.txt
│   │   dailydialog_test.txt
|   |   dailydialog_train.txt
|
```

### Training

#### Meld
------
```
python train.py -tr -wp 0 -bsz 1 -acc_step [2, 8] -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-[base, large] -epochs [20, 5]
```
#### Emorynlp
------
```
python train.py -tr -wp 0 -bsz 1 -acc_step [2, 8] -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path -tsk emorynlp roberta-[base, large] -epochs [20, 5]
```
#### Iemocap
------
```
python train_iemocap.py -tr -wp 0 -bsz 1 -acc_step [2, 8] -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-[base, large] -epochs [20, 5]
```
#### Daily Dialog
------
```
python train_dailydialog.py -tr -wp 0 -bsz 1 -acc_step [2, 8] -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-[base, large] -epochs [20, 5]
```

#### Without Speaker
-----
Similarly for training without the speaker information the training scripts `train_meld_emorynlp_without_speaker.py` and `train_iemocap_without_speaker.py`.

### For Inference
------
Please download the `model checkpoint` from the link given below and keep it in this directory. Following script will run only on a GPU.
```
python inference.py -tr -wp 0 -bsz 1 -acc_step 8 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-base -epochs 2 -postfix iemocap_inf -tsk iemocap -ft
```


### Checkpoint
------

| model                     | Checkpoint                                                   |
| ------------------------- | ------------------------------------------------------------ |
| Iemocap-roberta-base      | [iemocap-roberta-base.pkl](https://drive.google.com/file/d/154MIP5mUG7iwv0jhaN9iwfJ51wg8iU38/view?usp=sharing) |

Checkpoints are produced on a single A100 GPU.
