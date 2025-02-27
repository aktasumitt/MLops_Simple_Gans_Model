class Config:
    # Data Ingestion Stage
    train_data_path = "datasets/train"

    # Data Transformation Stage
    train_dataset_save_path = "artifacts/data_transformation/train_dataset.pkl"

    # Model Ingestion Stage
    generator_save_path = "artifacts/model/generator.pth"
    discriminator_save_path = "artifacts/model/discriminator.pth"

    # Training Stage
    checkpoint_save_path = "callbacks/checkpoints/checkpoint_latest.pth.tar"
    final_generator_model_path = "callbacks/final_model/generator_model.pth"
    final_discriminator_model_path = "callbacks/final_model/discriminator_model.pth"
    results_save_path = "results/train_results.json"

    # Testing Stage
    test_img_save_path = "results/test_images.jpg"
    test_model_path = "callbacks/final_model/generator_model.pth"

    # Prediction Stage
    predicted_img_save_path= "prediction_images/"
