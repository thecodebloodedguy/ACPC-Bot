from langdetect import detect
import spacy
from googletrans import Translator
import translators as ts

nlp = spacy.load('en_core_web_sm')

def isGuj(text):
    text_content = text.strip()
    
    # Detect the language of the input text
    detected_language = detect(text_content)

    
    # Compare the detected language with the desired language (Gujarati in this case)
    if detected_language == 'gu':
        return True
    else:
        return False


def translate(text):
    text=ts.translate_text(text)
    return text   

def en_input(text):
    if isGuj(text):
        return translate(text)
    else:
        return text


def toGuj(text):
    text=ts.translate_text(text)
    return text    




