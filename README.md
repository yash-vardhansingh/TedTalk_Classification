
# TedTalk Classification Flask Application

## Overview
The TedTalk Classification Flask Application leverages advanced natural language processing (NLP) techniques to classify TedTalks into various categories based on their transcripts. Utilizing a machine learning model trained on a diverse dataset, this application extracts transcripts from YouTube videos and predicts the most relevant categories, such as Education, Technology, Science, and more.

## Features
- **YouTube Transcript Extraction**: Automatically retrieves transcripts from TedTalk YouTube videos.
- **Text Classification**: Utilizes a pre-trained model to classify the extracted text into predefined categories.
- **BERT Model**: Employs BERT (Bidirectional Encoder Representations from Transformers) for understanding context and enhancing prediction accuracy.
- **User-Friendly Interface**: Offers a simple and intuitive web interface for users to submit YouTube URLs and receive classifications.

## Setup Instructions
1. **Clone the Repository**: 
    ```
    git clone <repository-url>
    ```
2. **Install Dependencies**: Navigate to the project directory and install required libraries using:
    ```
    pip install -r requirements.txt
    ```
3. **Model Setup**: Ensure the `exported_learner.pkl` file, containing the trained model, is placed in the project directory.

4. **Run the Application**:
    ```
    python app.py
    ```
    This command starts the Flask server.

5. **Access the Web Interface**: Open a browser and visit http://127.0.0.1:5000/.

## Usage
- Navigate to the web interface.
- Enter a TedTalk YouTube video URL into the provided field.
- Click on "Classify Text" to receive the talk's classification.

## How It Works
The application uses the YouTube Transcript API to extract video captions, which are then processed by a BERT-based model to classify the content into one of the several categories. This process involves understanding the context of the text, identifying key topics, and matching them to the most relevant category.

## Limitations
- Predictions rely on the availability and accuracy of YouTube video transcripts.
- The model's performance may vary based on the specificity of the content and the quality of the training data.

## Contributing
We welcome contributions to improve the TedTalk Classification Flask Application! Whether it's adding new features, improving the model, or fixing bugs, your help is appreciated. Please follow the standard fork-and-pull request workflow for contributions.

- **Fork the Repository**: Click on the "Fork" button at the top right of the page.
- **Clone Your Fork**: 
    ```
    git clone https://github.com/your-username/repository-name.git
    ```
- **Create a Pull Request**: After making your changes, push them to your fork and then submit a pull request.

## License
Specify your project's license here, providing users with details on how they can use, modify, and distribute your project.
