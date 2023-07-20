import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import data as dt

"""
Variable declarations:
"""
idx=0
flag=0
question=''
def get_answer_lines(text):
    text=int(text)
    if text==1:
        return dt.pharmacy_answers
    if text==2:
        return dt.mba_mca_answers
    if text==3:
        return dt.march_mplan_answers
    if text==4:
        return dt.mtech_answers
    else:
        return dt.be_answers
def get_lines(text):
    text=int(text)

    if text==1:
        return dt.pharmacy_questions
    if text==2:
        return dt.mba_mca_questions
    if text==3:
        return dt.march_mplan_questions
    if text==4:
        return dt.mtech_questions
answer_lines=[]
lines=[]


def find_best_match(input_text, lines):
    """
    Find the best match for the input text from the list of lines.

    Args:
        input_text (str): The input text to match.
        lines (list): A list of strings representing different lines.

    Returns:
        str: The best-matched line from the list.
    """
    # Tokenize the input text and the lines
    #nltk.download('punkt')
    input_tokens = nltk.word_tokenize(input_text)
    line_tokens = [nltk.word_tokenize(line) for line in lines]

    # Convert the tokens to lowercase
    input_tokens = [token.lower() for token in input_tokens]
    line_tokens = [[token.lower() for token in tokens] for tokens in line_tokens]

    # Join the tokens back to sentences for TF-IDF vectorization
    input_sentence = " ".join(input_tokens)
    line_sentences = [" ".join(tokens) for tokens in line_tokens]

    # Vectorize the input and lines using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([input_sentence] + line_sentences)
    input_vector, line_vectors = vectors[0], vectors[1:]

    # Calculate cosine similarities between the input and lines
    similarities = cosine_similarity(input_vector, line_vectors)

    # Find the best match and its similarity score
    best_match_idx = similarities.argmax()
    best_similarity_score = similarities[0][best_match_idx]

    # Check if the similarity score meets the threshold
    if best_similarity_score >= 0.25:
        best_match_line = lines[best_match_idx]
        for idx,xline in enumerate(lines):
            if best_match_line==xline:
                index = idx
                break
        return idx 
    else:
        if flag==0:
            return -1
        elif flag==1:
            return -2
        else:
            return -3
    
"""
Matching Functions for Answer

"""
def get_answer(question_index,answer_lines):

    answerline='Ask a valid Question'
    for idx,xline in enumerate(answer_lines):

        if idx==question_index:
            answerline=xline
           
    return answerline

def catselector(idx):
    if idx==-1:
        flag=1
        answer_lines=dt.be_answers
        lines=dt.be_questions
        idx=find_best_match(question,lines)

    if idx==-2:
        flag=2
        answer_lines=dt.greet_answers
        lines=dt.greet_questions
        idx=find_best_match(question,lines)
    if idx==-3:
        pass

    else:
        pass