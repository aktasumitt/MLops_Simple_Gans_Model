class Params:

    # Data Transformation Stage
    normalize_mean = (0.5,)
    normalize_std = (0.5,)

    # Model Ingestion Stage
    channel_size = 1
    label_size = 10
    embed_size = 64
    img_size = 32

    # Training Stage
    batch_size = 50
    noise_dim = 100
    device = "cuda"
    learning_rate = 0.001
    betas = (0.5, 0.99)
    epochs = 4
