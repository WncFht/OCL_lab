class Config:
    # 数据相关
    DATA_ROOT = '/data/DATA/OCL_DATA/OCL_data'
    BATCH_SIZE = 12
    NUM_WORKERS = 4
    TOP_K_CATEGORIES = 10
    
    # 模型相关
    MODEL_NAME = "google/siglip-base-patch16-224"
    
    # 日志相关w
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 输出相关
    SAVE_DIR = 'results'