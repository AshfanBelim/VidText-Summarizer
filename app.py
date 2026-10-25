"""AI-Powered Video Transcriber, Summarizer & Sentiment Analyzer (Gradio app)."""

import gc
import os
import shutil
import tempfile

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet Windows warning

import gradio as gr
import torch
import whisper
from docx import Document
from fpdf import FPDF
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

#config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lighter defaults on CPU-only laptops, heavier ones when a GPU is available.

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium" if DEVICE == "cuda" else "base")
T5_MODEL_NAME = os.getenv("T5_MODEL", "t5-large" if DEVICE == "cuda" else "t5-base")
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
CHUNK_WORDS = 300  # words per summarization chunk

if shutil.which("ffmpeg") is None:
    print(
        "WARNING: FFmpeg was not found on PATH. Video transcription will fail.\n"
        "Install it (Windows: 'winget install Gyan.FFmpeg', macOS: 'brew install ffmpeg', "
        "Linux: 'sudo apt install ffmpeg') and restart VS Code."
    )

# models
print(f"Using device: {DEVICE} | Whisper: {WHISPER_MODEL_SIZE} | T5: {T5_MODEL_NAME}")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    device=0 if DEVICE == "cuda" else -1,
)


# helpers
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summarize_chunk(chunk, max_length=150, min_length=30):
    inputs = t5_tokenizer(
        "summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        ids = t5_model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
    return t5_tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize_text(text):
    """Summarize text of any length by splitting it into chunks."""
    if not text or not text.strip():
        raise gr.Error("No text provided!")
    try:
        words = text.split()
        chunks = [
            " ".join(words[i : i + CHUNK_WORDS])
            for i in range(0, len(words), CHUNK_WORDS)
        ]
        parts = [_summarize_chunk(c) for c in chunks]
        return " ".join(parts)
    except Exception as e:
        raise gr.Error(f"Summarization error: {e}")


def transcribe_and_summarize(video_file):
    clear_memory()
    if not video_file:
        raise gr.Error("No video uploaded!")

    # Gradio may pass a path string or a file-like object depending on version
    path = video_file if isinstance(video_file, str) else video_file.name
    if shutil.which("ffmpeg") is None:
        raise gr.Error("FFmpeg is not installed or not on PATH. Install it and restart VS Code.")
    try:
        print(f"Processing file: {path}")
        # Whisper uses ffmpeg internally, so it reads video files directly
        result = whisper_model.transcribe(path, fp16=(DEVICE == "cuda"))
        transcript = result["text"].strip()
        if not transcript:
            raise gr.Error("No speech detected in the video.")
        summary = summarize_text(transcript)
        return transcript, summary
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Processing failed: {e}")


def analyze_sentiment(text):
    if not text or not text.strip():
        raise gr.Error("No text provided!")
    result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
    return f"**Sentiment:** {result['label']} (Confidence: {round(result['score'] * 100, 2)}%)"


def _output_path(filename):
    return os.path.join(tempfile.mkdtemp(), filename)


def save_as_pdf(text):
    if not text or not text.strip():
        raise gr.Error("No text to save!")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    # Built-in PDF fonts support Latin-1 only; unsupported characters become '?'
    safe_text = text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 10, safe_text)
    path = _output_path("output.pdf")
    pdf.output(path)
    return path


def save_as_word(text):
    if not text or not text.strip():
        raise gr.Error("No text to save!")
    doc = Document()
    doc.add_paragraph(text)
    path = _output_path("output.docx")
    doc.save(path)
    return path


#UI
with gr.Blocks(title="AI Summarizer & Video Transcription") as demo:
    gr.Markdown("# 📜 AI-Powered Summarizer & Video Transcription")

    with gr.Tab("📜 Text Summarization"):
        text_input = gr.Textbox(lines=5, placeholder="Enter text to summarize...", label="Input text")
        summarize_button = gr.Button("Summarize")
        text_summary_output = gr.Textbox(label="Summary")
        with gr.Row():
            pdf_button = gr.Button("Download as PDF")
            word_button = gr.Button("Download as Word")
        text_pdf_file = gr.File(label="PDF")
        text_word_file = gr.File(label="Word")

        summarize_button.click(summarize_text, text_input, text_summary_output)
        pdf_button.click(save_as_pdf, text_summary_output, text_pdf_file)
        word_button.click(save_as_word, text_summary_output, text_word_file)

    with gr.Tab("🎥 Video Transcription & Summarization"):
        video_input = gr.File(label="Upload video/audio", file_types=["video", "audio"])
        transcribe_button = gr.Button("Transcribe & Summarize")
        transcript_output = gr.Textbox(label="Full Transcript", lines=8)
        video_summary_output = gr.Textbox(label="Summarized Transcript", lines=6)

        with gr.Row():
            pdf_t = gr.Button("Transcript → PDF")
            word_t = gr.Button("Transcript → Word")
            pdf_s = gr.Button("Summary → PDF")
            word_s = gr.Button("Summary → Word")
        video_file_out = gr.File(label="Download")

        transcribe_button.click(
            transcribe_and_summarize, video_input, [transcript_output, video_summary_output]
        )
        pdf_t.click(save_as_pdf, transcript_output, video_file_out)
        word_t.click(save_as_word, transcript_output, video_file_out)
        pdf_s.click(save_as_pdf, video_summary_output, video_file_out)
        word_s.click(save_as_word, video_summary_output, video_file_out)

    with gr.Tab("🔹 Sentiment Analysis"):
        sentiment_input = gr.Textbox(lines=5, placeholder="Enter text for sentiment analysis...", label="Input text")
        analyze_button = gr.Button("Analyze Sentiment")
        sentiment_output = gr.Markdown()
        analyze_button.click(analyze_sentiment, sentiment_input, sentiment_output)


if __name__ == "__main__":
    # Opens http://127.0.0.1:7860 in your browser.
    # Set GRADIO_SHARE=1 to create a temporary public link.
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=os.getenv("GRADIO_SHARE") == "1",
    )