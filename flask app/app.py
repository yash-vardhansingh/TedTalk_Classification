from flask import Flask, request, render_template
from transformers import BartForConditionalGeneration, BartTokenizer
from fastai.text.all import load_learner
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

# Initialize global variables for summarization
summarization_model = None
tokenizer = None
# Your model's path
model = None
model_path = "./exported_learner.pkl"

def get_model():
    global model
    if model is None:
        model = load_learner(model_path)
    return model

def load_summarization_model():
    global summarization_model, tokenizer
    summarization_model = BartForConditionalGeneration.from_pretrained("./new")
    tokenizer = BartTokenizer.from_pretrained("./new")

def load_model():
    get_model()  # Load the Fastai model
    load_summarization_model()  # Load the BART model and tokenizer


def summarize_text(text_to_summarize):
    inputs = tokenizer(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/', methods=['GET'])
def index():
    # Render the HTML form, adjusted to accept YouTube URLs
    return render_template('index.html')

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

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        youtube_video_url = request.form.get('youtubeURL')
        transcript = extract_transcript_details(youtube_video_url)
        if transcript:
            # Perform classification
            model = get_model()
            pred_class, pred_idx, pred_probs = model.predict(transcript)
            top_preds = pred_probs.topk(2)
            top_classes = [model.dls.vocab[1][i] for i in top_preds.indices]
            top_probs = [round(p.item() * 100, 2) for p in top_preds.values]
            predictions = list(zip(top_classes, top_probs))

            # Perform summarization
            summary = summarize_text(transcript)

            return render_template('result.html', predictions=predictions, summary=summary)
        else:
            error_message = "Failed to extract transcript. Please check the YouTube URL and try again."
            return render_template('error.html', error_message=error_message)

load_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
