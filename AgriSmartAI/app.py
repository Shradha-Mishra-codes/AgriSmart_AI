import streamlit as st
import requests
from streamlit_lottie import st_lottie
from deep_translator import GoogleTranslator
import time
from geopy.geocoders import Nominatim
from PIL import Image
import json
from google import genai
from pydantic import BaseModel, Field
from gtts import gTTS
import base64
import io
import os
from fpdf import FPDF
from datetime import datetime
DB_FILE = "community_db.json"

def load_posts():
    """Safely loads posts from the JSON file."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_post(user_name, message, crop_type):
    """Saves a new post and handles file creation."""
    posts = load_posts()
    new_post = {
        "name": user_name,
        "crop": crop_type,
        "message": message,
        "time": datetime.now().strftime("%d %b, %I:%M %p")
    }
    posts.insert(0, new_post)
    with open(DB_FILE, "w") as f:
        json.dump(posts, f, indent=4)

# Define schema for Gemini Image Vision
class DiseaseAnalysis(BaseModel):
    is_healthy: bool = Field(description="True if the plant shows no signs of stress or disease.")
    health_percentage: int = Field(description="A score from 0-100 of the plant's current vitality.")
    name: str = Field(description="The specific name of the disease or 'Healthy Plant.'")
    explanation: str = Field(description="A brief, non-technical explanation of the visual symptoms.")
    step_by_step_cure: list[str] = Field(description="3 to 5 clear steps. USE THE TRAFFIC LIGHT SYSTEM: Start each string with '🔴 STOP:', '🟡 CAUTION:', or '🟢 GO:'.")
    traditional_remedies: list[str] = Field(description="2 to 3 Indian 'Gharelu Nuskhe' (e.g., Neem oil spray).")
    organic_chemical_fix: str = Field(description="One specific organic or bio-pesticide product name.")

class SoilAnalysis(BaseModel):
    nitrogen_status: str = Field(description="Low, Medium, or High based on card.")
    phosphorus_status: str = Field(description="Low, Medium, or High based on card.")
    potassium_status: str = Field(description="Low, Medium, or High based on card.")
    layman_summary: str = Field(description="A simple explanation of what the soil needs.")
    natural_fixes: list[str] = Field(description="Natural/organic ways to fix the deficiencies (e.g. plant Moong Dal to fix Nitrogen).")

class TimelineEvent(BaseModel):
    day: str = Field(description="e.g., 'Day 1-10'")
    action: str = Field(description="e.g., 'Sowing instructions'")

class CropRecommendation(BaseModel):
    recommended_crops: list[str] = Field(description="Top 3 crops recommended.")
    reasoning: str = Field(description="2-3 sentence reasoning based on Soil/Weather.")
    timeline: list[TimelineEvent] = Field(description="List of timeline events.")

st.set_page_config(page_title="AgriSmart AI", page_icon="🌾", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
if 'lang' not in st.session_state:
    st.session_state.lang = 'Hindi'
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ''

@st.cache_data(show_spinner=False)
def translate_text(text, dest_lang):
    if dest_lang == 'English' or not text:
        return text
    try:
        lang_map = {'Hindi': 'hi', 'English': 'en', 'Marathi': 'mr', 'Tamil': 'ta', 'Telugu': 'te', 'Odia': 'or', 'Gujarati': 'gu'}
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target=lang_map.get(dest_lang, 'hi'))
        if isinstance(text, list):
            return translator.translate_batch(text)
        else:
            return translator.translate(text)
    except Exception as e:
        return text  # fallback if translation fails

def get_base64_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def create_audio(text, dest_lang):
    try:
        lang_map = {'Hindi': 'hi', 'English': 'en', 'Marathi': 'mr', 'Tamil': 'ta', 'Telugu': 'te', 'Odia': 'or', 'Gujarati': 'gu'}
        lang_code = lang_map.get(dest_lang, 'hi') # fallback to Hindi
        tts = gTTS(text=text, lang=lang_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        return None

def transcribe_audio(audio_file, api_key):
    if not api_key: return ""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {'mime_type': 'audio/wav', 'data': audio_file.getvalue()}, 
                "You are an expert transcriptionist. Transcribe the audio exactly. Output only the text transcription in the same language as spoken. Output nothing else."
            ]
        )
        return response.text.strip()
    except Exception as e:
        return ""

def generate_pdf_report(data):
    lang = st.session_state.lang
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, translate_text("AgriSmart AI - Health Report", lang), ln=True)
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(0, 10, translate_text("Date", lang) + f": {datetime.now().strftime('%Y-%m-%d')}", ln=True)
        pdf.cell(0, 10, f"Health Vitality: {data.get('health_percentage', 0)}%", ln=True)
        pdf.cell(0, 10, f"Diagnosis: {data.get('name', 'N/A')}", ln=True)
        pdf.multi_cell(0, 10, f"Explanation: {data.get('explanation', '')}")
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, translate_text("Step-by-Step Cure", lang) + ":", ln=True)
        pdf.set_font("Arial", size=12)
        for idx, step in enumerate(data.get('step_by_step_cure', [])):
            safe_step = step.replace('🔴', '[STOP]').replace('🟡', '[CAUTION]').replace('🟢', '[GO]')
            safe_step = safe_step.encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 10, f"{idx+1}. {safe_step}")
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, translate_text("Organic / Chemical Fix", lang) + ":", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(data.get('organic_chemical_fix', '')))
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        return None

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def apply_custom_styles():
    primary_green = "#2E7D32" # Deep Green
    dark_text = "#1A1A1A"    # Near Black for maximum readability
    secondary_text = "#424242" # Dark Grey for descriptions
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        .stApp {{ 
            font-family: 'Outfit', sans-serif; 
            background-color: #fcfdfc; 
        }}

        /* --- Global Text Visibility Fix --- */
        h1, h2, h3, h4, p, span, label {{
            color:{dark_text} !important;
        }}

        /* --- Sidebar Specific Fix (Forces White Text) --- */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] label {{
            color: black !important;
        }}
        /* Fix for Sidebar Selectbox/Input labels specifically */
        [data-testid="stSidebar"] .stWidgetLabel p {{
            color: white !important;
        }}

        /* --- Hero Section (White text is okay here because background is dark) --- */
       .hero-section {{
            position: relative;
            height: 450px; 
            width: 100%;
            overflow: hidden;
            border-radius: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 40px;
            background-color: #000; 
        }}

        .video-wrapper video {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 450px;
            object-fit: cover;
            z-index: -1;
        }}
        
        .video-wrapper [data-testid="stVideoControls"] {{
            display: none;
        }}

        .video-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 450px;
            background: rgba(0, 0, 0, 0.4);
            z-index: 0;
        }}

        .hero-text-container {{
            position: relative;
            z-index: 1;
            padding: 100px 20px;
            text-align: center;
        }}

        .hero-section h1, .hero-section p, .hero-section span {{
            color: white !important; 
        }}

        .glass-card {{
            background: #ffffff; 
            border: 1px solid #e0e0e0;
            border-radius: 20px; 
            padding: 30px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            margin-bottom: 25px;
            color: {dark_text} !important;
        }}
        
        .glass-card h3 {{
            color: #2E7D32 !important;
            font-weight: 700;
        }}

        .glass-card p {{
            color: {secondary_text} !important;
            font-size: 1.05rem;
            line-height: 1.6;
        }}

        .stTabs [data-baseweb="tab"] p {{
            color: {secondary_text} !important;
            font-weight: 600 !important;
        }}
        
        .stTabs [aria-selected="true"] p {{
            color: {primary_green} !important;
        }}

        .stWidgetLabel p {{
            color: {dark_text} !important;
            font-weight: 600 !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_custom_styles()
# 2. THE LOGO SECTION (Must be BEFORE the menu)
# Using columns to force the logo to the center
col_l1, col_l2, col_l3 = st.columns([1, 0.8, 1])

with col_l2:
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        # Fallback centered text logo if image is missing
        st.markdown("""
            <h1 style='text-align: center; color: #2E7D32; margin-bottom: 0;'>🌱</h1>
            <h3 style='text-align: center; color: #2E7D32; margin-top: 0; font-weight: 800;'>AgriSmart AI</h3>
        """, unsafe_allow_html=True)

from streamlit_option_menu import option_menu
def t(text):
        return translate_text(text, st.session_state.lang)
selected = option_menu(
    menu_title=None,
    options=["Home", "About AgriSmart", "Feedback", "Login"],
    icons=["house", "book", "chat-dots", "person"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1A1A1A"},
        "icon": {"color": "#4CAF50", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#333"},
        "nav-link-selected": {"background-color": "#2E7D32"},
    }
)
if selected == "Home":
    # --- LOGO SECTION --
# Now show your navigation buttons below the logo
# selected = option_menu(None, ["Home", "About", ...], ...)
    hero_container = st.container()
    
    with hero_container:
        # 1. Video Path - ensure this file exists in assets/video.mp4
        video_path = 'assets/video.mp4' 
        
        if os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            bin_str = base64.b64encode(video_bytes).decode()
            
            # HTML for the Video Header
            st.markdown(f"""
                <div style='position: relative; width: 100%; height: 350px; border-radius: 25px; overflow: hidden; margin-bottom: 40px;'>
                    <video autoplay muted loop playsinline style='position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 1;'>
                        <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
                    </video>
                    <div style='position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.4); z-index: 2; display: flex; align-items: center; justify-content: center; text-align: center;'>
                        <div>
                            <h1 style='color: white !important;'>{translate_text("भविष्य के लिए खेती", st.session_state.lang)}</h1>
                            <p style='color: white !important;'>{translate_text("एग्रीस्मार्ट एआई: बुद्धिमान कृषि क्लाउड प्लेटफॉर्म।", st.session_state.lang)}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback Green Card if video is missing
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #2E7D32, #1b5e20); padding: 80px; border-radius: 25px; text-align: center; color: white; margin-bottom: 40px;'>
                    <h1 style='color: white !important;'>{translate_text("भविष्य के लिए खेती", st.session_state.lang)}</h1>
                    <p style='color: white !important;'>{translate_text("एग्रीस्मार्ट एआई: बुद्धिमान कृषि क्लाउड प्लेटफॉर्म।", st.session_state.lang)}</p>
                </div>
            """, unsafe_allow_html=True)

    # Tabs for the App
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🔍 {translate_text('Plant & Soil Analysis', st.session_state.lang)}", 
        f"💡 {translate_text('Smart Crop Advisor', st.session_state.lang)}",
        f"👨🏽‍🌾 {translate_text('Farmer Chat', st.session_state.lang)}",
        f"🤝 {translate_text('Community Board', st.session_state.lang)}"
    ])
    with tab1:
        st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
        header = t("Multi-Input Disease Detection")
        desc = t("Detect crop diseases instantly with our Deep AI. Upload an image, video, or take a realtime photo.")
        st.markdown(f"<h3>{header}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{desc}</p>", unsafe_allow_html=True)
        
        input_options = [
            f"📷 {t('Take a Photo')}", 
            f"🖼️ {t('Upload Image')}",
            f"📄 {t('Upload Soil Health Card')}"
        ]
        input_method = st.radio(t("Choose Input Method"), input_options, horizontal=True, label_visibility="collapsed")
        file = None
        if "📷" in input_method:
            file = st.camera_input(t("Take a picture"))
        elif "🖼️" in input_method or "📄" in input_method:
            file = st.file_uploader(t("Upload image"), type=["jpg", "jpeg", "png"])

        if file is not None:
            if not st.session_state.gemini_api_key:
                st.warning(t("⚠️ Please enter your Gemini API Key in the sidebar settings to unlock True AI Analysis."))
            else:
                with st.spinner(t("Analyzing Data with Gemini Vision...")):
                    try:
                        client = genai.Client(api_key=st.session_state.gemini_api_key)
                        img = Image.open(file)
                        is_soil = "📄" in input_method
                        
                        if is_soil:
                            prompt = "Role: AgriSmart AI Soil Analyst. Analyze the Soil Health Card. Extract N-P-K status. Provide a layman summary and natural fixes."
                            schema = SoilAnalysis
                        else:
                            prompt = "Role: AgriSmart AI Senior Plant Pathologist. Analyze crop disease and provide step-wise actionable advice."
                            schema = DiseaseAnalysis
                        
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[img, prompt],
                            config={'response_mime_type': 'application/json', 'response_schema': schema, 'temperature': 0.2}
                        )
                        data = json.loads(response.text)
                        st.success(t("✅ Analysis Complete!"))
                        
                        if is_soil:
                            st.markdown(f"#### 🧪 {t('Soil N-P-K Status')}")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Nitrogen (N)", t(data.get('nitrogen_status', 'Unknown')))
                            col2.metric("Phosphorus (P)", t(data.get('phosphorus_status', 'Unknown')))
                            col3.metric("Potassium (K)", t(data.get('potassium_status', 'Unknown')))
                            st.markdown(f"**{t('Summary:')}** {t(data.get('layman_summary', ''))}")
                            st.markdown(f"**{t('Natural Fixes:')}**")
                            for fix in data.get('natural_fixes', []):
                                st.write(f"🌱 {t(fix)}")
                        else:
                            h_percent = data.get('health_percentage', 0)
                            st.progress(h_percent / 100.0)
                            st.markdown(f"**{t('Health Vitality')}:** {h_percent}%")
                            audio_text = f"Diagnosis is {data.get('name', '')}. " + " ".join(data.get('step_by_step_cure', []))
                            audio_bytes = create_audio(t(audio_text), st.session_state.lang)
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.audio(audio_bytes)
                            with col_b:
                                pdf_bytes = generate_pdf_report(data)
                                if pdf_bytes:
                                    st.download_button(f"📥 {t('Save PDF')}", pdf_bytes, "agri_report.pdf")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>💡 {t('Smart Crop Advisor')}</h3>", unsafe_allow_html=True)
        location_container = st.container()
        with location_container:
            col_city1, col_city2 = st.columns([2, 1]) 
            with col_city1:
                city = st.text_input(
                    t("Enter your City/Location"),
                    placeholder="e.g., Mumbai, Maharashtra",
                    key="city_input_text"
                )
            with col_city2:
                st.write("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                city_audio = st.audio_input(
                    t("🎙️ Speak city"),
                    key="city_audio_input"
                )

        with st.form("crop_form", border=False):
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown(f"**{t('Soil Nutrients')}**")
                n_val = st.number_input("Nitrogen (N)", 0, 200, 50, help="Nitrogen content in soil")
                p_val = st.number_input("Phosphorus (P)", 0, 200, 50)
                k_val = st.number_input("Potassium (K)", 0, 200, 40)
            
            with c2:
                st.markdown(f"**{t('Environment')}**")
                temp_val = st.slider("Temp (°C)", 0.0, 50.0, 28.0)
                rain_val = st.slider("Rainfall (mm)", 0.0, 1000.0, 85.0)
            
            submit = st.form_submit_button(
                t("Get Recommendation"),
                use_container_width=True
            )
            
        if submit:
            with st.spinner(t("Calculating best crop for your soil...")):
                st.success(f"✅ {t('Recommended Crop: Rice or Cotton')}")
                st.balloons()
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>👨🏽‍🌾 {t('Farmer Chat')}</h3>", unsafe_allow_html=True)
        
        if "messages" not in st.session_state: 
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): 
                st.markdown(msg["content"])
            
        user_query = st.chat_input(t("Ask a question..."))
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})

    with tab4:
        st.markdown(f"### 🤝 {t('Community Board')}")
    
        with st.expander(f"📝 {t('Write a New Post')}"):
            with st.form("community_post_form", clear_on_submit=True):
                user_name = st.text_input(t("Your Name"), placeholder=t("Enter your name"))
                crop_name = st.text_input(t("Crop"), placeholder="e.g. Wheat, Tomato")
                user_msg = st.text_area(t("Message"), placeholder=t("Share your farming update..."))
                
                submit_post = st.form_submit_button(t("Post Now"))
                
                if submit_post:
                    if user_name and user_msg:
                        save_post(user_name, user_msg, crop_name)
                        st.success(t("Post Published!"))
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(t("Please provide both a Name and a Message."))

        st.markdown("---")
        
        all_posts = load_posts()
        if not all_posts:
            st.info(t("No posts yet. Be the first to share an update!"))
        else:
            for p in all_posts:
                st.markdown(f"""
                    <div style="background-color: white; padding: 15px; border-radius: 15px; 
                                border-left: 5px solid #2E7D32; box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
                                margin-bottom: 15px;">
                        <p style="color: #666; font-size: 0.8rem; margin-bottom: 5px;">{p['time']}</p>
                        <strong style="color: #2E7D32; font-size: 1.1rem;">{p['name']}</strong> 
                        <span style="color: #555;"> • {t('Growing')}: {p['crop'] if p['crop'] else t('General')}</span>
                        <p style="color: #333; margin-top: 10px; line-height: 1.5;">{p['message']}</p>
                    </div>
                """, unsafe_allow_html=True)

elif selected == "About AgriSmart":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.title(t("📖 About AgriSmart AI"))
    st.markdown(t("AgriSmart AI is a next-generation platform designed to empower farmers with Multimodal AI. Features include disease detection, soil analysis, crop recommendations, farmer chat, and community board."))
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Feedback":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.title(t("📝 Farmer Feedback"))
    with st.form("feedback_form"):
        name = st.text_input(t("Name"))
        rating = st.slider(t("Rate us"), 1, 5, 5)
        comment = st.text_area(t("Improvement suggestions?"))
        if st.form_submit_button(t("Submit")):
            st.success(t("Thank you for your feedback!"))
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Login":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.title(t("🔐 Farmer Login"))
    email = st.text_input(t("Email"))
    password = st.text_input(t("Password"), type="password")
    if st.button(t("Login")):
        st.info(t("Login successful! (Demo mode)"))
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h2 style='text-align: center; color: white;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    # Use 'key' to ensure language state is tracked correctly
    # In your sidebar code:
    st.selectbox("Language", ["English", "Hindi", "Marathi", "Gujarati"], key='lang')
    st.text_input("Gemini API Key", type="password", key='gemini_api_key')
    
    lottie_plant = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_tbg14xow.json")
    if lottie_plant:
        st_lottie(lottie_plant, height=150)
    st.markdown("<p style='text-align:center; color: white;'>AgriSmart AI v2.0</p>", unsafe_allow_html=True)