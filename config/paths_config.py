import os

########################## data ingestion ################
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR,"raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR,"train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR,"test.csv")

CONFIG_PATH = "config/config.yaml"


################### data processing step ############

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR,"processed_train.csv")
PROCESSED_TEST_FILE_PATH = os.path.join(PROCESSED_DIR,"processed_test.csv")

############## Model training session ######
MODEL_OUTPUT_PATH = "artifact/models/lgbm_model.pkl"