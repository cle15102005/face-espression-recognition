# Face Emotion Recognition Project

## Models:
- SVM, RF, SRC, XGB: in `main_train.py` (default)
- CNN: in `CNN_main_train.py`
- Tuning option only available for RF, XGB and SRC. For SVM, the highest parameters are chosen after tuning before.

## Run Instructions:

### 1. Build Docker Image
docker build -t face-emotion-app .
### 2. Dataset
- All datasets are at: https://husteduvn-my.sharepoint.com/:f:/g/personal/linh_nn235599_sis_hust_edu_vn/EiTklgue2T1Cg_DBjXN9tbEBUIw3MXkIGgEkiT3c9n846A?e=VJpME0
- For CNN, download the folder "images_balanced" and move it into this project folder, check the paths in "CNN_main_train.py" 
if error occur.
- For SVM, RF, SRC, XGB, download "emotional_train_limited.csv" and "emotional_val_limited.csv" and move it into this project folder, check the paths in "main_train.py" if error occur.
### 2. Run models
- For SVM, RF, SRC, XGB: docker run -it face-emotion-app
- For CNN: docker run -it face-emotion-app python CNN_main_train.py
