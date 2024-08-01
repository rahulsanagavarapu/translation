import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration, BartTokenizer
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
import os
import tempfile
from langdetect import detect, LangDetectException
import PyPDF2
import io
import time
import jieba
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu import corpus_bleu

nltk.download('punkt')

@st.cache_resource
def load_models_and_tokenizers():
    # Chinese to English model
    zh_en_tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model/tokenizer')
    zh_en_model = AutoModelForSeq2SeqLM.from_pretrained('fine_tuned_model/model')
    
    # English to Chinese model
    en_zh_tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    en_zh_model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    
    summarization_model = BartForConditionalGeneration.from_pretrained('philschmid/bart-large-cnn-samsum')
    summarization_tokenizer = BartTokenizer.from_pretrained('philschmid/bart-large-cnn-samsum')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zh_en_model.to(device)
    en_zh_model.to(device)
    summarization_model.to(device)

    return zh_en_model, zh_en_tokenizer, en_zh_model, en_zh_tokenizer, summarization_model, summarization_tokenizer, device

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

def translate_chinese_to_english(chinese_text, model, tokenizer, device):
    sentences = chinese_text.split('。')
    translated_sentences = []
    confidence_scores = []

    for sentence in sentences:
        if sentence.strip():
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=12,
                    length_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            translation = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            translated_sentences.append(translation)
            
            scores = outputs.sequences_scores
            confidence_score = torch.exp(scores).item()
            confidence_scores.append(confidence_score)

    full_translation = ' '.join(translated_sentences)
    average_confidence = sum(confidence_scores) / len(confidence_scores)
    return full_translation, average_confidence

def back_translate(english_text, model, tokenizer, device):
    sentences = english_text.split('. ')
    back_translated_sentences = []

    for sentence in sentences:
        if sentence.strip():
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=12,
                    length_penalty=1.2,
                    early_stopping=True
                )
            back_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            back_translated_sentences.append(back_translation)

    return '。'.join(back_translated_sentences)

def calculate_bleu_score(reference, candidate):
    def char_tokenize(text):
        return list(text.replace(" ", ""))

    reference_tokens = [char_tokenize(reference)]
    candidate_tokens = char_tokenize(candidate)

    weights = (0.25, 0.25, 0.25, 0.25)  # Use equal weights for 1-gram to 4-gram
    smoothing_function = SmoothingFunction().method4  # Using method4 for smoother results

    return sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smoothing_function)

def calculate_sacrebleu_score(reference, candidate):
    return corpus_bleu([candidate], [[reference]]).score

def sliding_window_bleu(reference, candidate, window_size=100, step_size=50):
    ref_chars = list(reference.replace(" ", ""))
    cand_chars = list(candidate.replace(" ", ""))
    
    scores = []
    for i in range(0, len(ref_chars) - window_size + 1, step_size):
        ref_window = ''.join(ref_chars[i:i+window_size])
        cand_window = ''.join(cand_chars[i:i+window_size])
        scores.append(calculate_bleu_score(ref_window, cand_window))
    
    return sum(scores) / len(scores) if scores else 0

def summarize_text(text, model, tokenizer, device, max_length=1500, min_length=2):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

def speech_to_text():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Speak now...")
            audio = r.listen(source, timeout=30, phrase_time_limit=30)
            st.write("Processing speech...")
        
        text = r.recognize_google(audio, language="zh-CN")
        return text
    except sr.UnknownValueError:
        st.error("Speech recognition could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from speech recognition service; {e}")
    except Exception as e:
        st.error(f"An error occurred during speech recognition: {str(e)}")
    return None

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def translate_summarize_and_display(text, zh_en_model, zh_en_tokenizer, en_zh_model, en_zh_tokenizer, summarization_model, summarization_tokenizer, device, input_type):
    # Detect language
    detected_lang = detect_language(text)
    if detected_lang != 'zh-cn':
        st.error(f"Error: The input text appears to be in {detected_lang}. This app only supports Chinese to English translation.")
        return

    # Translation
    translation_start = time.time()
    with st.spinner("Translating..."):
        english_translation, confidence_score = translate_chinese_to_english(text, zh_en_model, zh_en_tokenizer, device)
    translation_end = time.time()
    translation_time = translation_end - translation_start
    st.success(f"Translation complete! Time taken: {translation_time:.2f} seconds")
    st.info(f"Translation confidence score: {confidence_score:.2%}")

    # Back-translation
    back_translation_start = time.time()
    with st.spinner("Performing back-translation..."):
        back_translated_text = back_translate(english_translation, en_zh_model, en_zh_tokenizer, device)
    back_translation_end = time.time()
    back_translation_time = back_translation_end - back_translation_start
    st.success(f"Back-translation complete! Time taken: {back_translation_time:.2f} seconds")

    # Calculate BLEU scores
    char_bleu_score = calculate_bleu_score(text, back_translated_text)
    sacrebleu_score = calculate_sacrebleu_score(text, back_translated_text)
    sliding_bleu_score = sliding_window_bleu(text, back_translated_text)

    st.info(f"Character-level BLEU score: {char_bleu_score:.2%}")
    st.info(f"SacreBLEU score: {sacrebleu_score:.2f}")
    st.info(f"Sliding window BLEU score: {sliding_bleu_score:.2%}")

    if input_type in ["text", "speech"]:
        st.write("Full English translation:")
        st.write(english_translation)
        
        st.write("Back-translated Chinese text:")
        st.write(back_translated_text)
        
        # Generate audio for full translation
        audio_start = time.time()
        with st.spinner("Generating audio for full translation..."):
            full_audio_file = text_to_speech(english_translation)
        audio_end = time.time()
        audio_time = audio_end - audio_start
        st.success(f"Full translation audio generated! Time taken: {audio_time:.2f} seconds")
        st.subheader("Full Translation Audio:")
        st.audio(full_audio_file)
        os.unlink(full_audio_file)

    # Summarization
    summary_start = time.time()
    with st.spinner("Generating summary..."):
        summary = summarize_text(english_translation, summarization_model, summarization_tokenizer, device)
    summary_end = time.time()
    summary_time = summary_end - summary_start
    st.success(f"Summary generated! Time taken: {summary_time:.2f} seconds")
    st.write("Summary:")
    st.write(summary)
    
    # Generate audio for summary
    summary_audio_start = time.time()
    with st.spinner("Generating audio for summary..."):
        summary_audio_file = text_to_speech(summary)
    summary_audio_end = time.time()
    summary_audio_time = summary_audio_end - summary_audio_start
    st.success(f"Summary audio generated! Time taken: {summary_audio_time:.2f} seconds")
    st.subheader("Summary Audio:")
    st.audio(summary_audio_file)
    os.unlink(summary_audio_file)

    # Total time
    total_time = translation_time + back_translation_time + summary_time
    if input_type in ["text", "speech"]:
        total_time += audio_time
    total_time += summary_audio_time
    st.info(f"Total processing time: {total_time:.2f} seconds")

def main():
    st.title("Chinese to English Translation App with PDF Upload, Speech-to-Text, and Summarization")

    zh_en_model, zh_en_tokenizer, en_zh_model, en_zh_tokenizer, summarization_model, summarization_tokenizer, device = load_models_and_tokenizers()

    st.header("Input")
    input_method = st.radio("Choose input method:", ("Text", "Speech", "PDF Upload"))

    chinese_text = None

    if input_method == "Text":
        chinese_text = st.text_area("Enter Chinese text to translate (use '。' to separate sentences):")
        if st.button("Translate and Summarize"):
            if not chinese_text:
                st.error("Please provide input text before translating.")
                return
            translate_summarize_and_display(chinese_text, zh_en_model, zh_en_tokenizer, en_zh_model, en_zh_tokenizer, summarization_model, summarization_tokenizer, device, "text")
    elif input_method == "Speech":
        st.write("Note: Make sure your microphone is connected and working.")
        if st.button("Start Recording"):
            with st.spinner("Listening..."):
                chinese_text = speech_to_text()
            if chinese_text:
                st.success("Speech recognized!")
                st.write("Recognized text:", chinese_text)
                translate_summarize_and_display(chinese_text, zh_en_model, zh_en_tokenizer, en_zh_model, en_zh_tokenizer, summarization_model, summarization_tokenizer, device, "speech")
            else:
                st.error("Failed to recognize speech. Please check your microphone and try again.")
                return
    else:  # PDF Upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
            st.success("Text extracted from PDF!")
            if pdf_text:
                translate_summarize_and_display(pdf_text, zh_en_model, zh_en_tokenizer, en_zh_model, en_zh_tokenizer, summarization_model, summarization_tokenizer, device, "pdf")
            else:
                st.error("No text could be extracted from the PDF. Please ensure it contains readable text.")

if __name__ == "__main__":
    main()