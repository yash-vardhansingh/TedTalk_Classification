from flask import Flask, request, render_template
from fastai.text.all import load_learner
from pathlib import Path, WindowsPath, PosixPath
import pathlib
from contextlib import contextmanager
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

# Define the context manager to temporarily patch PosixPath
@contextmanager
def set_posix_to_windows_path():
    original_posix_path = PosixPath
    try:
        pathlib.PosixPath = WindowsPath
        yield
    finally:
        pathlib.PosixPath = original_posix_path

model = None
# Your model's path
model_path = Path(r".\exported_learner.pkl")

@app.route('/', methods=['GET'])
def index():
    # Render the HTML form, adjusted to accept YouTube URLs
    return render_template('index.html')

@app.before_first_request
def load_model():
    global model
    with set_posix_to_windows_path():
        model = load_learner(model_path)

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split('v=')[1]
        # Handling URLs with additional parameters by splitting on '&'
        video_id = video_id.split('&')[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([item['text'] for item in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error extracting transcript: {e}")
        return None
"""
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        youtube_video_url = request.form.get('youtubeURL')
        transcript = extract_transcript_details(youtube_video_url)
        if transcript:
            pred_class, _, _ = model.predict(transcript)
            return render_template('result.html', predicted_class=pred_class)
        else:
            error_message = "Failed to extract transcript. Please check the YouTube URL and try again."
            return render_template('error.html', error_message=error_message)
"""

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        youtube_video_url = request.form.get('youtubeURL')
        transcript = extract_transcript_details(youtube_video_url)
        if transcript:
            pred_class, pred_idx, pred_probs = model.predict(transcript)
            # Get the top 2 predictions and their probabilities
            top_preds = pred_probs.topk(2)
            top_classes = [model.dls.vocab[1][i] for i in top_preds.indices]
            top_probs = [round(p.item() * 100, 2) for p in top_preds.values]
            # Pack the top predictions and their probabilities in a list of tuples
            predictions = list(zip(top_classes, top_probs))
            return render_template('result.html', predictions=predictions)
        else:
            error_message = "Failed to extract transcript. Please check the YouTube URL and try again."
            return render_template('error.html', error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
