import glob
SEED = 2018
INPUT_SHAPE = (112, 112, 3)
NUM_CLASSES = 1000
TOP_5 = 'top_k_categorical_accuracy'
TOP_1 = 'categorical_accuracy'
MODEL_ARCH_TABLE = 'model_arch_library'
MODEL_SELECTION_TABLE = 'mst_table'
MODEL_SELECTION_SUMMARY_TABLE = 'mst_table_summary'

param_grid = {
    "learning_rate": [1e-4, 1e-6],
    "lambda_value": [1e-4, 1e-6],
    "batch_size": [32, 256],
    "model": ["vgg16", "resnet50"]
}
param_grid_hetro = {
    "learning_rate": [1e-4],
    "lambda_value": [1e-4],
    "batch_size": [2, 16, 32, 64, 128, 256],
    "model": ["resnet50", "mobilenetv2", "nasnetmobile", "vgg16"]
}
param_grid_scalability = {
    "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
    "lambda_value": [1e-4, 1e-6],
    "batch_size": [32],
    "model": ["resnet50"]
}
param_grid_model_size = {
    's': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["mobilenetv2"]
    },
    'm': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["resnet50"]
    },
    'l': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["resnet152"]
    },
    'x': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["vgg16"]
    },
}
param_grid_best_model = {
    "learning_rate": [1e-4],
    "lambda_value": [1e-4],
    "batch_size": [32],
    "model": ["resnet50"]
}
