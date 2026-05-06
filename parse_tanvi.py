import PyPDF2
import sys

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

try:
    text = extract_text_from_pdf("Tanvi.pdf")
    print(text[:1000]) # Print first 1000 characters
except Exception as e:
    print(f"Error: {e}")
