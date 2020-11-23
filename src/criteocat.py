INPUT_SHAPE = (7306, )
NUM_CLASSES = 2

param_grid_criteo = {
    "learning_rate": [1e-3, 1e-4],
    "lambda_value": [1e-4, 1e-5],
    "batch_size": [32, 64, 256, 512],
    "model": ["confA"]
}

param_grid_criteo_breakdown = {
    "learning_rate": [1e-3, 1e-4],
    "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
    "batch_size": [256],
    "model": ["confA"]
}