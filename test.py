import streamlit as st
from google.oauth2 import service_account
from google.cloud import speech
import io
import torch
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2 import Wav2Vec2Model
import librosa
from groq import Groq


Groq_api_key= st.secrets["GROQ_API_KEY"]

google_creds = st.secrets["google_creds"]

# Initialize Google Speech-to-Text, Hugging Face model, and Groq LLM
# client_file = "gcp_api.json"  # Replace with your actual service account file path
credentials = service_account.Credentials.from_service_account_info(google_creds)
speech_client = speech.SpeechClient(credentials=credentials)

processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("models/wav2vec2-base")

ideal_embedding = torch.tensor(np.load("ideal_embedding_part_1.npy"))
Groq_api_key= st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=Groq_api_key)

# Define the ideal Azan text (first part only) and its English meaning
ideal_text = "اللّٰهُ أَكْبَرُ، اللّٰهُ أَكْبَرُ"
ideal_text_meaning = "Allah is the Greatest, Allah is the Greatest"

# Function to extract embedding of the uploaded audio
def get_audio_embedding(audio_file_path):
    audio_input, _ = librosa.load(audio_file_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model(inputs.input_values).last_hidden_state.mean(dim=1).squeeze()
    return embedding

# Function to calculate similarity score between user and ideal embeddings
def calculate_similarity(embedding1, embedding2):
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
    return similarity.item() * 100

# Function to generate feedback with Groq LLM
def generate_feedback_with_llm(user_transcription, ideal_text, similarity_score): 
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert muezzin trainer providing detailed, supportive feedback on a student's Azan transcription accuracy. "
                "The student has attempted to recite the phrase, and their transcription is compared to the ideal Azan phrase in Arabic. "
                "Evaluate how closely their recitation matches the ideal Azan text based on articulation, tone, rhythm, and accuracy. "
                "Emphasize strengths, point out specific areas where improvements can be made, and give clear, practical tips to improve pronunciation. "
                "Encourage the student with positivity, helping them refine their pronunciation and accuracy until it aligns closely with the ideal."
            )
        },
        {
            "role": "user",
            "content": (
                f"The student's transcription of their recitation is: '{user_transcription}'. "
                f"The ideal phrase for comparison is: '{ideal_text}'. Their similarity score is {similarity_score:.2f}%. "
                "Please provide feedback highlighting strengths, improvement areas, and actionable tips for better alignment with the ideal."
            )
        }
    ]
    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return completion.choices[0].message.content

# Function to transcribe audio, validate with the ideal text, and provide feedback
def transcribe_and_validate(audio_file_path, ideal_text):
    with io.open(audio_file_path, 'rb') as f:
        audio_content = f.read()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=48000,
        language_code="ar"
    )
    response = speech_client.recognize(config=config, audio=audio)
    transcription = " ".join(result.alternatives[0].transcript for result in response.results)

    # Refined prompt for validation with LLM
    content = f"""
        You are an expert in validating the Azaan (the call to prayer). Below is the correct structure of the Azaan. 
        Compare the transcription provided with this structure to determine if it contains all essential phrases in the correct order.
        Validation Guidelines:
        - Validate the Azaan as "VALIDATED" if it contains all essential phrases in the correct sequence, even if there are minor spelling, diacritic, or punctuation differences.
        - Specifically, ignore small differences such as:
            - Missing or extra diacritics (e.g., "ا" vs. "أ" or "حي على الصلاه" vs. "حي على الصلاة").
            - Minor spelling variations, such as:
                - "لا اله الا الله" vs. "لا إله إلا الله".
                - "حي على الصلاه" vs. "حي على الصلاة".
                - "حي على الفلاح" vs. "حي على الفلاح".
                - "أشهد" vs "شهاده"
            - Punctuation or slight variations in commonly understood words and phrases.
        - Invalidate the Azaan as "INVALIDATED" only if:
            - Essential phrases are missing.
            - Extra, unrelated phrases that are not part of the Azaan are added.
            - Major incorrect words or substitutions that change the meaning of an essential phrase are present.
        Correct Azaan Structure:
        "{ideal_text}"
        Transcribed Azaan:
        "{transcription}"
        Conclude with "Validation Status: VALIDATED" if the Azaan matches the correct structure, or "Validation Status: INVALIDATED" if it does not, and list any specific issues if found. Only list issues if they involve missing phrases, extra phrases, or significant meaning changes.
    """

    # Send request to Groq LLM for validation feedback
    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=512,
    )
    feedback = completion.choices[0].message.content

    return transcription, feedback

# Streamlit layout
st.set_page_config(page_title="Azan Pronunciation Trainer", layout="centered", initial_sidebar_state="expanded")

# Display ideal text and its meaning for reference
st.markdown("<div style='font-size: 30px; color: #4CAF50; text-align: center;'>Azan Pronunciation Trainer</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: 20px; text-align: center;'>Phrase to Practice: {ideal_text}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: 18px; text-align: center; color: #555;'>Meaning: {ideal_text_meaning}</div>", unsafe_allow_html=True)

# Placeholder for expert audio playback
st.audio("expert_azan_audio.mp3", format="audio/mp3")  # Replace with actual path

# Upload audio file for pronunciation assessment
st.markdown("<div style='font-size: 18px; text-align: center;'>Upload your Azan recitation audio (MP3 format):</div>", unsafe_allow_html=True)
audio_file = st.file_uploader("Choose an audio file", type=["mp3"])

if audio_file is not None:
    with st.spinner("Analyzing your pronunciation..."):
        audio_path = "uploaded_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        # Transcribe and validate transcription with the ideal text
        transcription, validation_feedback = transcribe_and_validate(audio_path, ideal_text)

        # Check if validation is successful
        if "Validation Status: VALIDATED" in validation_feedback:
            # Perform similarity check if validated
            user_embedding = get_audio_embedding(audio_path)
            similarity_score = calculate_similarity(user_embedding, ideal_embedding)
            
            st.markdown(f"<div style='font-size: 18px; color: #333; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;'><b>Similarity Score:</b> {similarity_score:.2f}%</div>", unsafe_allow_html=True)

            # Provide feedback based on similarity score
            if similarity_score >= 90:
                st.markdown("<div style='color: green; font-weight: bold;'>Excellent work! Your pronunciation is reverent and accurate. You may proceed to the next phrase.</div>", unsafe_allow_html=True)
            else:
                llm_feedback = generate_feedback_with_llm(transcription, ideal_text, similarity_score)
                st.markdown(f"<div style='font-size: 18px; color: #333; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;'><b>Feedback:</b><br>{llm_feedback}</div>", unsafe_allow_html=True)
        else:
            # Inform user to re-record if validation failed
            st.markdown("<div style='color: red; font-weight: bold;'>The transcription does not match the ideal Azan phrase. Please record your recitation again.</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 18px; color: #333; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;'><b>Feedback:</b><br>{validation_feedback}</div>", unsafe_allow_html=True)
