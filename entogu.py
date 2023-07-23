# import Levenshtein
# from translation_data import gujarati_sentence_list
# def toEng(english_sentence):
    
#     best_distance = float('inf')
#     best_match = None

#     for gujarati_sentence in gujarati_sentence_list:
#         distance = Levenshtein.distance(english_sentence, gujarati_sentence)

#         if distance < best_distance:
#             best_distance = distance
#             best_match = gujarati_sentence

#     return best_match

from deep_translator import GoogleTranslator
def toEng(text):
    return GoogleTranslator(source='auto', target='gu').translate(text) 
