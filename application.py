from src.pipeline.prediction_pipeline import PredictionPipeline
from flask import Flask, request, render_template, send_from_directory
import os

app = Flask(__name__)
predictor = PredictionPipeline()
config=predictor.prediction_config

# Flask'ın tahmin edilen resimlerin bulunduğu klasörü sunmasını sağlıyoruz
@app.route('/<filename>')
def send_prediction_image(filename):
    return send_from_directory(config.predicted_img_save_path, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_paths = []
    if request.method == 'POST':
        try:
            filenames = predictor.run_prediction_pipeline()  # Tahminleri al
            image_paths = [os.path.basename(f) for f in filenames]  # URL'leri oluştur
        
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html', image_paths=image_paths)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

