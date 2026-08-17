import nltk
import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag


# ============================================================
# 1. SPAM WORD DICTIONARY
# ============================================================
# You can add your own words and risk levels here.
# Format: ["word", "risk"]

spam_words = [
    ["free", "High"],
    ["win", "High"],
    ["winner", "High"],
    ["prize", "High"],
    ["urgent", "Low"],
    ["click", "High"],
    ["offer", "High"],
    ["offers", "High"],
    ["discount", "Medium"],
    ["bonus", "Medium"],
    ["cash", "Medium"],
    ["money", "Medium"],
    ["congratulations", "Medium"],
    ["sale", "Low"],
    ["deal", "Low"],
    ["limited", "Medium"],
    ["buy", "Low"],
    ["exciting", "Medium"],
]


# ============================================================
# 2. STEP 1 - CASING
# ============================================================
def convert_to_lowercase(sentence):
    return sentence.lower()


# ============================================================
# 3. STEP 2 - TOKENIZATION
# ============================================================
def tokenize(sentence):
    return word_tokenize(sentence)


# ============================================================
# 4. STEP 3 - STOP WORD / POS REMOVAL
# ============================================================
stop_words = set(stopwords.words('english'))

def remove_unwanted_words(tokens):
    tagged_words = pos_tag(tokens)
    filtered_words = []

    for word, tag in tagged_words:

        # ----------------------------------------------------
        # KEEP CURRENCY SYMBOLS
        # ----------------------------------------------------
        if word in ["$", "€", "£", "₹", "¥"]:
            filtered_words.append(word)
            continue

        # ----------------------------------------------------
        # KEEP WORDS PRESENT IN SPAM DICTIONARY
        # ----------------------------------------------------
        spam_word_list = [item[0] for item in spam_words]

        if word in spam_word_list:
            filtered_words.append(word)
            continue

        # ----------------------------------------------------
        # REMOVE SPECIAL CHARACTERS
        # ----------------------------------------------------
        if all(char in string.punctuation for char in word):
            continue

        # ----------------------------------------------------
        # REMOVE STOP WORDS
        # ----------------------------------------------------
        if word in stop_words:
            continue

        # ----------------------------------------------------
        # REMOVE ADJECTIVES, ADVERBS AND VERBS
        # ----------------------------------------------------
        if tag.startswith(("JJ", "RB", "VB")):
            continue

        # ----------------------------------------------------
        # KEEP EVERYTHING ELSE
        # ----------------------------------------------------
        filtered_words.append(word)

    return filtered_words


# ============================================================
# 5. STEP 4 - ANALYSIS
# ============================================================
def analyse_words(words):
    analysis_dictionary = []
    for word in words:
        risk = "Low"
        # Search our spam-word dictionary
        for spam_word, spam_risk in spam_words:
            if word == spam_word:
                risk = spam_risk
                break
        analysis_dictionary.append([word, risk])
    return analysis_dictionary


# ============================================================
# 6. STEP 5 - SPAM DETECTION
# ============================================================
def detect_spam(analysis_dictionary):
    if len(analysis_dictionary) == 0:
        return "Not Spam"

    high_count = 0
    medium_count = 0
    low_count = 0

    total_words = len(analysis_dictionary)

    for word, risk in analysis_dictionary:

        if risk == "High":
            high_count += 1

        elif risk == "Medium":
            medium_count += 1

        elif risk == "Low":
            low_count += 1

    # Calculate percentages
    high_percentage = (high_count / total_words) * 100
    medium_percentage = (medium_count / total_words) * 100
    low_percentage = (low_count / total_words) * 100
    # --------------------------------------------------------
    # YOUR SPAM CONDITION
    # --------------------------------------------------------
    # You can change these conditions later.
    
    if high_percentage >= 60:
        result = "SPAM"

    elif high_percentage >= 30 and medium_percentage >= 30:
        result = "SPAM"

    elif high_percentage >= 20 and medium_percentage >= 20 and low_percentage >= 20:
        result = "SPAM"

    elif medium_percentage >= 80:
            result = "SPAM"

    elif medium_percentage >= 60 and low_percentage >= 30:
            result = "SPAM"

    else:
        result = "NOT SPAM"

    return result


# ============================================================
# 7. TAKE 5 INPUTS
# ============================================================

for i in range(5):

    print("\n========================================")
    print("Input", i + 1)
    print("========================================")

    sentence = input("Enter a sentence: ")

    # STEP 1 - Casing
    lowercase_sentence = convert_to_lowercase(sentence)
    # STEP 2 - Tokenization
    tokens = tokenize(lowercase_sentence)
    # STEP 3 - Remove unwanted words
    filtered_words = remove_unwanted_words(tokens)
    # STEP 4 - Analyse words
    analysis = analyse_words(filtered_words)
    # STEP 5 - Detect spam
    result = detect_spam(analysis)

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nOriginal Sentence:")
    print(sentence)

    print("\nAfter Casing:")
    print(lowercase_sentence)

    print("\nTokens:")
    print(tokens)

    print("\nAfter Stop Word / POS Removal:")
    print(filtered_words)

    print("\nWord Risk Analysis:")
    print(analysis)

    print("\nFinal Result:")
    print(result)