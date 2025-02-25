from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictionPipeline

# Flask Uygulaması
app = Flask(__name__)
predictor = PredictionPipeline()

@app.route('/', methods=['GET', 'POST'])
def index():
    image_paths = []
    if request.method == 'POST':
        try:
            image_paths = predictor.run_prediction_pipeline()  # Random tahmin
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html', image_paths=image_paths)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

