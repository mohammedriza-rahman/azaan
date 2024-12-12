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
import sounddevice as sd
import scipy.io.wavfile as wav
import os
from datetime import datetime
from pydub import AudioSegment
from pathlib import Path
from openai import OpenAI
import json
import plotly.graph_objects as go

# Enhanced UI Styles
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Base styles */
:root {
    --primary-color: #2563eb;
    --secondary-color: #1d4ed8;
    --success-color: #059669;
    --warning-color: #d97706;
    --danger-color: #dc2626;
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --bg-primary: #ffffff;
    --bg-secondary: #f3f4f6;
}

.stApp {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
    background: var(--bg-secondary);
}

/* Header styles */
.app-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    padding: 2rem 1rem;
    text-align: center;
    border-radius: 0 0 1.5rem 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.app-title {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.app-subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.2rem;
    font-weight: 500;
    direction: rtl;
}

/* Card styles */
.card {
    background: var(--bg-primary);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--bg-secondary);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

/* Button styles */
.button-container {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}

.button-primary {
    background-color: var(--primary-color);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    border: none;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.button-primary:hover {
    background-color: var(--secondary-color);
}

.button-danger {
    background-color: var(--danger-color);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    border: none;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease;
}

/* Progress indicator */
.score-container {
    text-align: center;
    padding: 1.5rem;
    background: var(--bg-secondary);
    border-radius: 1rem;
    margin-bottom: 1.5rem;
}

.score-value {
    font-size: 3rem;
    font-weight: 700;
    color: var(--primary-color);
}

.score-label {
    color: var(--text-secondary);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Feedback section */
.feedback-section {
    background: var(--bg-secondary);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.feedback-item {
    background: white;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Status messages */
.success-msg {
    background-color: var(--success-color);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: slideIn 0.3s ease;
}

.error-msg {
    background-color: var(--danger-color);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: slideIn 0.3s ease;
}

/* Animations */
@keyframes slideIn {
    from { transform: translateY(-10px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .app-title {
        font-size: 2rem;
    }
    
    .card {
        padding: 1rem;
    }
    
    .button-container {
        flex-direction: column;
    }
    
    .score-value {
        font-size: 2.5rem;
    }
}
</style>
"""


class AzanTrainerApp:
    def __init__(self):
        self.setup_api_clients()
        self.setup_configs()
        self.setup_directories()
        self.initialize_models()

    def setup_api_clients(self):
        """Initialize API clients"""
        self.openai_client = OpenAI(api_key=OpenAI_api_key)
        self.groq_client = Groq(api_key=Groq_api_key)
        self.speech_client = self.init_google_speech()

    def init_google_speech(self):
        """Initialize Google Speech client"""
        credentials = service_account.Credentials.from_service_account_file(
            "sa_speecch_demo.json"
        )
        return speech.SpeechClient(credentials=credentials)

    def setup_configs(self):
        """Set up configuration variables"""
        self.SAMPLE_RATE = 48000
        self.DURATION = 6
        self.AUDIO_GAIN = 1.50
        self.IDEAL_TEXT = "اللّٰهُ أَكْبَرُ، اللّٰهُ أَكْبَرُ"
        self.IDEAL_TEXT_MEANING = "Allah is the Greatest, Allah is the Greatest"

    def setup_directories(self):
        """Create necessary directories"""
        for dir_name in ['recordings', 'feedback_audio']:
            os.makedirs(dir_name, exist_ok=True)

    def initialize_models(self):
        """Initialize ML models"""
        self.processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("models/wav2vec2-base")
        self.ideal_embedding = torch.tensor(np.load("ideal_embedding_part_1.npy"))

    def create_waveform_visualization(self, audio_path, reference_path):
        """Create waveform visualization using Plotly"""
        fig = go.Figure()

        # Process user audio
        y_user, sr_user = librosa.load(audio_path)
        times_user = np.arange(len(y_user)) / sr_user
        fig.add_trace(go.Scatter(
            x=times_user,
            y=y_user,
            name='Your Recording',
            line=dict(color='#1E88E5')
        ))

        # Process reference audio
        y_ref, sr_ref = librosa.load(reference_path)
        times_ref = np.arange(len(y_ref)) / sr_ref
        fig.add_trace(go.Scatter(
            x=times_ref,
            y=y_ref,
            name='Expert Recording',
            line=dict(color='#4CAF50')
        ))

        fig.update_layout(
            title='Waveform Comparison',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            template='plotly_white',
            height=400
        )

        return fig

    def record_audio(self):
        """Record audio from user"""
        try:
            audio_data = sd.rec(
                int(self.DURATION * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype=np.float32
            )
            sd.wait()
            return self.enhance_audio(audio_data)
        except Exception as e:
            st.error(f"Recording error: {str(e)}")
            return None

    def enhance_audio(self, audio_data):
        """Enhance audio quality"""
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        audio_data = audio_data * self.AUDIO_GAIN
        noise_threshold = 0.01
        audio_data[np.abs(audio_data) < noise_threshold] = 0
        return audio_data

    def save_audio(self, audio_data):
        """Save audio to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/audio_{timestamp}.wav"
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        wav.write(filename, self.SAMPLE_RATE, audio_data)
        return filename

    def analyze_recording(self, audio_path):
        """Analyze the recording"""
        try:
            # Convert to MP3 for Google Speech API
            mp3_path = audio_path.replace('.wav', '.mp3')
            AudioSegment.from_wav(audio_path).export(mp3_path, format="mp3")

            # Transcribe audio
            with open(mp3_path, 'rb') as f:
                content = f.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code="ar"
            )

            response = self.speech_client.recognize(config=config, audio=audio)
            transcription = " ".join(result.alternatives[0].transcript 
                                   for result in response.results)

            # Calculate similarity
            user_embedding = self.get_audio_embedding(audio_path)
            similarity_score = self.calculate_similarity(user_embedding, self.ideal_embedding)

            # Generate feedback
            feedback = self.generate_feedback(transcription, similarity_score)

            # Clean up
            os.remove(mp3_path)

            return transcription, similarity_score, feedback

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return None, None, None

    def get_audio_embedding(self, audio_path):
        """Generate audio embedding"""
        audio_input, _ = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio_input, sampling_rate=16000, 
                              return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.model(inputs.input_values).last_hidden_state.mean(dim=1).squeeze()
        return embedding

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate similarity score"""
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        return similarity.item() * 100

    def generate_feedback(self, transcription, similarity_score):
        """Generate feedback in natural Roman Urdu using LLM"""
        prompt = f"""
        Is Azan ki tilawat ka jaiza len aur natural Roman Urdu main feedback den:

        Tilawat: {transcription}
        Mutabiqat Score: {similarity_score:.2f}%

        Feedback ko in 3 hisson main takseem karen:

        1. Talaffuz (Pronunciation):
        - Har lafz ka talaffuz kaisa hai
        - Huroof ki tartib theek hai ya nahi
        - Allah ke lafz ka talaffuz kaisa hai
        - Mukammal Azan ki tarteeb kaisi hai

        2. Waqt aur Lehja (Timing):
        - Har hissay ka sahi dohrao
        - Waqfay ki durustagi
        - Aawaz ka utaar chadhao

        3. Behtar Karne Ke Liye Mashwaray:
        - Kahan ghaltiyan hain
        - Kya behtar karna hai
        - Kis cheez par zyada mehnat ki zaroorat hai

        Note: Feedback zabaan-e-urdu main likhen, lekin English huroof istimal karen. 
        Lehja mohtaram aur madadgaar hona chahiye.
        """

        response = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def generate_audio_feedback(self, feedback_text):
        """Generate audio feedback"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = f"feedback_audio/feedback_{timestamp}.mp3"

            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=feedback_text
            )

            response.stream_to_file(audio_path)
            return audio_path

        except Exception as e:
            st.error(f"Error generating audio feedback: {str(e)}")
            return None

    def run(self):
        """Run the enhanced Streamlit application with Persian/Masjid-inspired UI"""
        st.set_page_config(
            page_title="Azan Pronunciation Trainer",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Custom CSS with Persian/Masjid-inspired theme (Keep your existing CSS here)
        st.markdown("""
            <style>
            /* Global Styles */
            @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
            
            :root {
                --primary-color: #1F4C6B;
                --secondary-color: #C3934B;
                --accent-color: #E6B17E;
                --background-color: #F7F3E9;
                --text-color: #2C3E50;
                --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .stApp {
                background-color: var(--background-color);
                font-family: 'Amiri', serif;
            }

            /* Header Styles */
            .app-header {
                background: linear-gradient(135deg, var(--primary-color), #2C3E50);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: var(--card-shadow);
            }

            .app-title {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                font-weight: 700;
                background: linear-gradient(45deg, var(--accent-color), #FFD700);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .app-subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                margin: 0.5rem 0;
            }

            .arabic-text {
                font-family: 'Amiri', serif;
                font-size: 2rem;
                direction: rtl;
                margin: 1rem 0;
                color: var(--secondary-color);
            }

            /* Card Styles */
            .card {
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: var(--card-shadow);
                border: 1px solid rgba(195, 147, 75, 0.2);
                transition: transform 0.2s ease;
            }

            .card:hover {
                transform: translateY(-2px);
            }

            .card-header {
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
                border-bottom: 2px solid var(--accent-color);
                padding-bottom: 0.5rem;
            }

            .card-title {
                font-size: 1.3rem;
                margin: 0 0 0 0.5rem;
                color: var(--primary-color);
            }

            /* Button Styles */
            .stButton button {
                background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                font-weight: bold;
                transition: all 0.3s ease;
                width: 100%;
                margin: 0.5rem 0;
            }

            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(31, 76, 107, 0.2);
            }

            /* Score Display */
            .score-container {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                margin: 1.5rem 0;
            }

            .score-value {
                font-size: 3rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }

            .score-label {
                font-size: 1.2rem;
                opacity: 0.9;
            }

            /* Feedback Styles */
            .feedback-item {
                background-color: rgba(195, 147, 75, 0.1);
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 4px solid var(--secondary-color);
            }

            /* Help Section Styling */
            .help-container {
                background: white;
                padding: 1.5rem;
                border-radius: 15px;
                margin-top: 1rem;
            }

            .help-item {
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
                padding: 0.5rem;
                border-radius: 8px;
                background-color: rgba(31, 76, 107, 0.05);
            }

            .help-number {
                background-color: var(--primary-color);
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 1rem;
                font-size: 0.9rem;
            }
            </style>
        """, unsafe_allow_html=True)

        # Enhanced Header with Arabic Styling
        st.markdown(f"""
            <div class="app-header">
                <h1 class="app-title">Azan Pronunciation Trainer</h1>
                <p class="app-subtitle">Perfect Your Recitation</p>
                <div class="arabic-text">{self.IDEAL_TEXT}</div>
                <p class="app-subtitle">{self.IDEAL_TEXT_MEANING}</p>
            </div>
        """, unsafe_allow_html=True)

        # Expert demonstration card
        st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span style="font-size: 2rem;">📹</span>
                    <h2 class="card-title">Expert Demonstration</h2>
                </div>
            """, unsafe_allow_html=True)
        st.video("qari part-1.mp4")
        st.markdown("</div>", unsafe_allow_html=True)

        # Expert audio card
        st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span style="font-size: 2rem;">🎵</span>
                    <h2 class="card-title">Reference Audio</h2>
                </div>
            """, unsafe_allow_html=True)
        st.audio("qari_part_1.mp3")
        st.markdown("</div>", unsafe_allow_html=True)

        # Recording controls card
        st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span style="font-size: 2rem;">🎙️</span>
                    <h2 class="card-title">Recording Controls</h2>
                </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Recording", help="Click to start recording (6 seconds)", key="start_rec"):
                with st.spinner("Recording in progress..."):
                    audio_data = self.record_audio()
                    if audio_data is not None:
                        audio_path = self.save_audio(audio_data)
                        st.session_state['audio_file'] = audio_path
                        st.markdown("""
                            <div class="feedback-item" style="background-color: rgba(46, 204, 113, 0.1); border-left-color: #2ecc71;">
                                Recording completed successfully! ✅
                            </div>
                        """, unsafe_allow_html=True)

        with col2:
            if st.button("Clear Recording", key="clear_rec"):
                if 'audio_file' in st.session_state:
                    if os.path.exists(st.session_state['audio_file']):
                        os.remove(st.session_state['audio_file'])
                    st.session_state['audio_file'] = None
                    st.markdown("""
                        <div class="feedback-item" style="background-color: rgba(231, 76, 60, 0.1); border-left-color: #e74c3c;">
                            Recording cleared! 🗑️
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Analysis section
        if 'audio_file' in st.session_state and st.session_state['audio_file']:
            st.markdown("""
                <div class="card">
                    <div class="card-header">
                        <span style="font-size: 2rem;">🎵</span>
                        <h2 class="card-title">Your Recording</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            st.audio(st.session_state['audio_file'])

            if st.button("Analyze Recording", key="analyze"):
                with st.spinner("Analyzing your recitation..."):
                    transcription, similarity, feedback = self.analyze_recording(
                        st.session_state['audio_file']
                    )

                    if all([transcription, similarity, feedback]):
                        # Enhanced similarity score display
                        st.markdown(f"""
                            <div class="score-container">
                                <div class="score-value">{similarity:.1f}%</div>
                                <div class="score-label">Similarity Score</div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Waveform visualization
                        fig = self.create_waveform_visualization(
                            st.session_state['audio_file'],
                            "qari_part_1.mp3"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Feedback display
                        st.markdown(f"""
                            <div class="card">
                                <div class="card-header">
                                    <span style="font-size: 2rem;">📝</span>
                                    <h2 class="card-title">Detailed Feedback</h2>
                                </div>
                                <div class="feedback-item">
                                    {feedback}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Audio feedback
                        audio_feedback_path = self.generate_audio_feedback(feedback)
                        if audio_feedback_path:
                            st.markdown("""
                                <div class="card">
                                    <div class="card-header">
                                        <span style="font-size: 2rem;">🔊</span>
                                        <h2 class="card-title">Audio Feedback</h2>
                                    </div>
                                """, unsafe_allow_html=True)
                            st.audio(audio_feedback_path)
                            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced help section with numbered steps
        with st.expander("❓ How to Use"):
            st.markdown("""
                <div class="help-container">
                    <div class="help-item">
                        <div class="help-number">1</div>
                        <div>Watch the expert demonstration video carefully</div>
                    </div>
                    <div class="help-item">
                        <div class="help-number">2</div>
                        <div>Listen to the reference audio to understand proper pronunciation</div>
                    </div>
                    <div class="help-item">
                        <div class="help-number">3</div>
                        <div>Click 'Start Recording' and recite the phrase (6 seconds)</div>
                    </div>
                    <div class="help-item">
                        <div class="help-number">4</div>
                        <div>Wait for the recording to complete</div>
                    </div>
                    <div class="help-item">
                        <div class="help-number">5</div>
                        <div>Click 'Analyze Recording' to get detailed feedback</div>
                    </div>
                    <div class="help-item">
                        <div class="help-number">6</div>
                        <div>Review your score and feedback to improve</div>
                    </div>
                    <div class="help-item">
                        <div class="help-number">7</div>
                        <div>Practice until you achieve 90% or higher similarity</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = AzanTrainerApp()
    app.run()