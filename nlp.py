# ============================================================
# ADVANCED NLP SPAM DETECTION AND BAG OF WORDS ANALYSIS SYSTEM
# ============================================================
#
# Features:
# 1. Text preprocessing
# 2. Tokenization
# 3. Stop-word removal
# 4. POS-based filtering
# 5. Extensive spam dictionary
# 6. Spam word categorization
# 7. Spam phrase detection
# 8. URL detection
# 9. Email detection
# 10. Phone number detection
# 11. Currency detection
# 12. Capital letter analysis
# 13. Punctuation analysis
# 14. Repeated word analysis
# 15. Weighted spam scoring
# 16. Structured word analysis
# 17. Bag of Words
# 18. Binary Bag of Words
# 19. TF-IDF
# 20. N-Gram analysis
# 21. Document-Term Matrix
# 22. Visualization dashboard
#
# Required packages:
#
# pip install nltk matplotlib numpy scikit-learn
#
# NLTK downloads:
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("averaged_perceptron_tagger_eng")
#
# ============================================================

import nltk
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import Counter, defaultdict

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

# ============================================================
# NLTK RESOURCE DOWNLOAD
# ============================================================

def download_nltk_resources():

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng"
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ============================================================
# EXTENSIVE SPAM DICTIONARY
# ============================================================
#
# Structure:
#
# "word": {
#     "risk": "High/Medium/Low",
#     "score": numerical_weight,
#     "category": category_name
# }
#
# ============================================================

SPAM_DICTIONARY = {

    # --------------------------------------------------------
    # FREE / MONEY / REWARD WORDS
    # --------------------------------------------------------

    "free": {
        "risk": "High",
        "score": 10,
        "category": "Free Offer"
    },

    "cash": {
        "risk": "Medium",
        "score": 7,
        "category": "Money"
    },

    "money": {
        "risk": "Medium",
        "score": 7,
        "category": "Money"
    },

    "reward": {
        "risk": "High",
        "score": 9,
        "category": "Reward"
    },

    "rewards": {
        "risk": "High",
        "score": 9,
        "category": "Reward"
    },

    "bonus": {
        "risk": "Medium",
        "score": 6,
        "category": "Bonus"
    },

    "prize": {
        "risk": "High",
        "score": 10,
        "category": "Prize"
    },

    "prizes": {
        "risk": "High",
        "score": 10,
        "category": "Prize"
    },

    "jackpot": {
        "risk": "High",
        "score": 10,
        "category": "Gambling / Reward"
    },

    "wealth": {
        "risk": "Medium",
        "score": 6,
        "category": "Money"
    },

    "income": {
        "risk": "Low",
        "score": 3,
        "category": "Financial"
    },

    "profit": {
        "risk": "Medium",
        "score": 6,
        "category": "Financial"
    },

    "earn": {
        "risk": "Medium",
        "score": 6,
        "category": "Financial"
    },

    "earning": {
        "risk": "Medium",
        "score": 6,
        "category": "Financial"
    },

    "earnings": {
        "risk": "Medium",
        "score": 6,
        "category": "Financial"
    },

    "million": {
        "risk": "High",
        "score": 8,
        "category": "Money"
    },

    "billion": {
        "risk": "High",
        "score": 8,
        "category": "Money"
    },

    # --------------------------------------------------------
    # WINNING / LOTTERY
    # --------------------------------------------------------

    "win": {
        "risk": "High",
        "score": 10,
        "category": "Lottery"
    },

    "winner": {
        "risk": "High",
        "score": 10,
        "category": "Lottery"
    },

    "winning": {
        "risk": "High",
        "score": 10,
        "category": "Lottery"
    },

    "lottery": {
        "risk": "High",
        "score": 10,
        "category": "Lottery"
    },

    "lucky": {
        "risk": "Medium",
        "score": 6,
        "category": "Lottery"
    },

    "selected": {
        "risk": "Medium",
        "score": 5,
        "category": "Selection"
    },

    "chosen": {
        "risk": "Medium",
        "score": 5,
        "category": "Selection"
    },

    "congratulations": {
        "risk": "Medium",
        "score": 7,
        "category": "Reward"
    },

    "congratulation": {
        "risk": "Medium",
        "score": 7,
        "category": "Reward"
    },

    # --------------------------------------------------------
    # URGENCY / PRESSURE
    # --------------------------------------------------------

    "urgent": {
        "risk": "High",
        "score": 9,
        "category": "Urgency"
    },

    "immediately": {
        "risk": "High",
        "score": 8,
        "category": "Urgency"
    },

    "immediate": {
        "risk": "High",
        "score": 8,
        "category": "Urgency"
    },

    "hurry": {
        "risk": "Medium",
        "score": 7,
        "category": "Urgency"
    },

    "quick": {
        "risk": "Low",
        "score": 3,
        "category": "Urgency"
    },

    "fast": {
        "risk": "Low",
        "score": 3,
        "category": "Urgency"
    },

    "now": {
        "risk": "Low",
        "score": 2,
        "category": "Urgency"
    },

    "today": {
        "risk": "Low",
        "score": 2,
        "category": "Urgency"
    },

    "deadline": {
        "risk": "Medium",
        "score": 5,
        "category": "Urgency"
    },

    "expire": {
        "risk": "High",
        "score": 7,
        "category": "Urgency"
    },

    "expired": {
        "risk": "Medium",
        "score": 6,
        "category": "Urgency"
    },

    # --------------------------------------------------------
    # CLICK / LINK / ACTION
    # --------------------------------------------------------

    "click": {
        "risk": "High",
        "score": 9,
        "category": "Suspicious Action"
    },

    "link": {
        "risk": "Medium",
        "score": 5,
        "category": "Link"
    },

    "visit": {
        "risk": "Low",
        "score": 3,
        "category": "Action"
    },

    "subscribe": {
        "risk": "Medium",
        "score": 5,
        "category": "Action"
    },

    "unsubscribe": {
        "risk": "Low",
        "score": 2,
        "category": "Email"
    },

    "download": {
        "risk": "Medium",
        "score": 5,
        "category": "Download"
    },

    "install": {
        "risk": "Medium",
        "score": 5,
        "category": "Download"
    },

    "register": {
        "risk": "Medium",
        "score": 4,
        "category": "Registration"
    },

    "signup": {
        "risk": "Medium",
        "score": 5,
        "category": "Registration"
    },

    "login": {
        "risk": "Medium",
        "score": 5,
        "category": "Credential"
    },

    # --------------------------------------------------------
    # OFFERS / SALES
    # --------------------------------------------------------

    "offer": {
        "risk": "High",
        "score": 8,
        "category": "Offer"
    },

    "offers": {
        "risk": "High",
        "score": 8,
        "category": "Offer"
    },

    "discount": {
        "risk": "Medium",
        "score": 6,
        "category": "Discount"
    },

    "sale": {
        "risk": "Low",
        "score": 4,
        "category": "Sale"
    },

    "deal": {
        "risk": "Low",
        "score": 4,
        "category": "Deal"
    },

    "deals": {
        "risk": "Low",
        "score": 4,
        "category": "Deal"
    },

    "limited": {
        "risk": "Medium",
        "score": 6,
        "category": "Scarcity"
    },

    "exclusive": {
        "risk": "Medium",
        "score": 6,
        "category": "Marketing"
    },

    "special": {
        "risk": "Low",
        "score": 3,
        "category": "Marketing"
    },

    "cheap": {
        "risk": "Medium",
        "score": 5,
        "category": "Offer"
    },

    "lowest": {
        "risk": "Medium",
        "score": 5,
        "category": "Offer"
    },

    "guaranteed": {
        "risk": "High",
        "score": 9,
        "category": "Guarantee"
    },

    # --------------------------------------------------------
    # ACCOUNT / SECURITY / CREDENTIAL PHISHING
    # --------------------------------------------------------

    "account": {
        "risk": "Low",
        "score": 2,
        "category": "Account"
    },

    "password": {
        "risk": "High",
        "score": 8,
        "category": "Credential"
    },

    "username": {
        "risk": "Medium",
        "score": 6,
        "category": "Credential"
    },

    "verify": {
        "risk": "High",
        "score": 8,
        "category": "Verification"
    },

    "verification": {
        "risk": "High",
        "score": 8,
        "category": "Verification"
    },

    "confirm": {
        "risk": "Medium",
        "score": 6,
        "category": "Verification"
    },

    "security": {
        "risk": "Medium",
        "score": 5,
        "category": "Security"
    },

    "suspended": {
        "risk": "High",
        "score": 9,
        "category": "Account Threat"
    },

    "blocked": {
        "risk": "High",
        "score": 8,
        "category": "Account Threat"
    },

    "locked": {
        "risk": "High",
        "score": 8,
        "category": "Account Threat"
    },

    "unauthorized": {
        "risk": "High",
        "score": 8,
        "category": "Security"
    },

    # --------------------------------------------------------
    # FINANCIAL / BANKING
    # --------------------------------------------------------

    "bank": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "banking": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "credit": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "debit": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "payment": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "transaction": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "refund": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "investment": {
        "risk": "Medium",
        "score": 5,
        "category": "Financial"
    },

    "crypto": {
        "risk": "High",
        "score": 8,
        "category": "Cryptocurrency"
    },

    "bitcoin": {
        "risk": "High",
        "score": 8,
        "category": "Cryptocurrency"
    },

    # --------------------------------------------------------
    # SUSPICIOUS ACTIONS
    # --------------------------------------------------------

    "claim": {
        "risk": "High",
        "score": 9,
        "category": "Suspicious Action"
    },

    "redeem": {
        "risk": "High",
        "score": 8,
        "category": "Reward Action"
    },

    "act": {
        "risk": "Low",
        "score": 3,
        "category": "Action"
    },

    "respond": {
        "risk": "Medium",
        "score": 5,
        "category": "Action"
    },

    "reply": {
        "risk": "Medium",
        "score": 5,
        "category": "Action"
    },

    "contact": {
        "risk": "Low",
        "score": 3,
        "category": "Action"
    },

    "call": {
        "risk": "Medium",
        "score": 5,
        "category": "Action"
    },

    # --------------------------------------------------------
    # MARKETING / PROMOTIONAL
    # --------------------------------------------------------

    "amazing": {
        "risk": "Medium",
        "score": 5,
        "category": "Marketing"
    },

    "exciting": {
        "risk": "Medium",
        "score": 5,
        "category": "Marketing"
    },

    "incredible": {
        "risk": "Medium",
        "score": 5,
        "category": "Marketing"
    },

    "fantastic": {
        "risk": "Medium",
        "score": 5,
        "category": "Marketing"
    },

    "best": {
        "risk": "Low",
        "score": 3,
        "category": "Marketing"
    },

    "perfect": {
        "risk": "Medium",
        "score": 4,
        "category": "Marketing"
    },

    "opportunity": {
        "risk": "Medium",
        "score": 5,
        "category": "Marketing"
    },

    # --------------------------------------------------------
    # SCAM / FRAUD RELATED
    # --------------------------------------------------------

    "riskfree": {
        "risk": "High",
        "score": 9,
        "category": "Fraud Pattern"
    },

    "guarantee": {
        "risk": "High",
        "score": 8,
        "category": "Guarantee"
    },

    "guarantees": {
        "risk": "High",
        "score": 8,
        "category": "Guarantee"
    },

    "no-risk": {
        "risk": "High",
        "score": 8,
        "category": "Fraud Pattern"
    },

    "miracle": {
        "risk": "High",
        "score": 7,
        "category": "Fraud Pattern"
    },

    "secret": {
        "risk": "Medium",
        "score": 5,
        "category": "Fraud Pattern"
    },

    "confidential": {
        "risk": "Medium",
        "score": 5,
        "category": "Fraud Pattern"
    },

    "winner": {
        "risk": "High",
        "score": 10,
        "category": "Lottery"
    }
}

# ============================================================
# EXTENSIVE SPAM PHRASE DICTIONARY
# ============================================================

SPAM_PHRASES = {

    "click here": {
        "risk": "High",
        "score": 12,
        "category": "Suspicious Link"
    },

    "click now": {
        "risk": "High",
        "score": 12,
        "category": "Urgent Action"
    },

    "click below": {
        "risk": "High",
        "score": 11,
        "category": "Suspicious Link"
    },

    "limited time": {
        "risk": "High",
        "score": 10,
        "category": "Urgency"
    },

    "limited offer": {
        "risk": "High",
        "score": 10,
        "category": "Scarcity"
    },

    "act now": {
        "risk": "High",
        "score": 12,
        "category": "Urgency"
    },

    "claim now": {
        "risk": "High",
        "score": 12,
        "category": "Reward"
    },

    "claim your prize": {
        "risk": "High",
        "score": 15,
        "category": "Lottery"
    },

    "you have won": {
        "risk": "High",
        "score": 15,
        "category": "Lottery"
    },

    "you are selected": {
        "risk": "High",
        "score": 12,
        "category": "Selection Scam"
    },

    "congratulations you": {
        "risk": "High",
        "score": 10,
        "category": "Reward"
    },

    "free money": {
        "risk": "High",
        "score": 15,
        "category": "Financial Scam"
    },

    "earn money": {
        "risk": "High",
        "score": 12,
        "category": "Financial Scam"
    },

    "make money": {
        "risk": "High",
        "score": 12,
        "category": "Financial Scam"
    },

    "work from home": {
        "risk": "Medium",
        "score": 7,
        "category": "Employment Spam"
    },

    "urgent response": {
        "risk": "High",
        "score": 12,
        "category": "Urgency"
    },

    "verify your account": {
        "risk": "High",
        "score": 15,
        "category": "Phishing"
    },

    "confirm your account": {
        "risk": "High",
        "score": 15,
        "category": "Phishing"
    },

    "account suspended": {
        "risk": "High",
        "score": 15,
        "category": "Account Threat"
    },

    "account locked": {
        "risk": "High",
        "score": 15,
        "category": "Account Threat"
    },

    "update your information": {
        "risk": "High",
        "score": 12,
        "category": "Phishing"
    },

    "password verification": {
        "risk": "High",
        "score": 15,
        "category": "Credential Phishing"
    },

    "risk free": {
        "risk": "High",
        "score": 10,
        "category": "Fraud Pattern"
    },

    "money back": {
        "risk": "Medium",
        "score": 7,
        "category": "Financial"
    },

    "buy now": {
        "risk": "Medium",
        "score": 7,
        "category": "Marketing"
    },

    "special offer": {
        "risk": "Medium",
        "score": 7,
        "category": "Marketing"
    },

    "exclusive offer": {
        "risk": "High",
        "score": 10,
        "category": "Marketing"
    },

    "instant cash": {
        "risk": "High",
        "score": 12,
        "category": "Financial Scam"
    },

    "easy money": {
        "risk": "High",
        "score": 12,
        "category": "Financial Scam"
    },

    "guaranteed income": {
        "risk": "High",
        "score": 12,
        "category": "Financial Scam"
    }
}

# ============================================================
# STOP WORDS
# ============================================================

stop_words = set(stopwords.words("english"))

# ============================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================

def convert_to_lowercase(text):

    return text.lower()

# ------------------------------------------------------------

def tokenize_words(sentence):

    return word_tokenize(sentence)

# ------------------------------------------------------------

def tokenize_sentences(paragraph):

    return sent_tokenize(paragraph)

# ------------------------------------------------------------

def remove_punctuation(tokens):

    return [
        word
        for word in tokens
        if not all(char in string.punctuation for char in word)
    ]

# ------------------------------------------------------------

def remove_stopwords(tokens):

    return [
        word
        for word in tokens
        if word.lower() not in stop_words
    ]

# ------------------------------------------------------------

def remove_unwanted_words(tokens):

    tagged_words = pos_tag(tokens)

    filtered_words = []

    for word, tag in tagged_words:

        word_lower = word.lower()

        # Keep currency symbols

        if word in ["$", "â‚¬", "Â£", "â‚¹", "Â¥"]:

            filtered_words.append(word)
            continue

        # Keep words present in spam dictionary

        if word_lower in SPAM_DICTIONARY:

            filtered_words.append(word_lower)
            continue

        # Remove punctuation

        if all(char in string.punctuation for char in word):

            continue

        # Remove stop words

        if word_lower in stop_words:

            continue

        # Remove selected POS categories

        if tag.startswith(("JJ", "RB")):

            continue

        filtered_words.append(word_lower)

    return filtered_words

# ============================================================
# SPECIAL PATTERN DETECTION
# ============================================================

def detect_urls(text):

    pattern = r"(https?://[^\s]+|www\.[^\s]+)"

    urls = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    return urls

# ------------------------------------------------------------

def detect_emails(text):

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    emails = re.findall(pattern, text)

    return emails

# ------------------------------------------------------------

def detect_phone_numbers(text):

    pattern = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{6,10}"

    phone_numbers = re.findall(pattern, text)

    return phone_numbers

# ------------------------------------------------------------

def detect_currency(text):

    pattern = r"[$â‚¬Â£â‚¹Â¥]\s?\d+(?:,\d{3})*(?:\.\d+)?"

    currency_values = re.findall(pattern, text)

    return currency_values

# ------------------------------------------------------------

def detect_excessive_punctuation(text):

    pattern = r"[!?.]{3,}"

    matches = re.findall(pattern, text)

    return matches

# ------------------------------------------------------------

def calculate_capital_ratio(text):

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if len(letters) == 0:

        return 0

    uppercase_letters = [
        char
        for char in letters
        if char.isupper()
    ]

    ratio = (
        len(uppercase_letters)
        /
        len(letters)
    ) * 100

    return ratio

# ------------------------------------------------------------

def detect_repeated_words(tokens):

    frequency = Counter(tokens)

    repeated = {
        word: count
        for word, count in frequency.items()
        if count >= 3
    }

    return repeated

# ============================================================
# SPAM WORD ANALYSIS
# ============================================================

def analyse_words(words):

    analysis_dictionary = []

    for word in words:

        word_lower = word.lower()

        if word_lower in SPAM_DICTIONARY:

            information = SPAM_DICTIONARY[word_lower]

            analysis_dictionary.append({

                "word": word,

                "risk": information["risk"],

                "score": information["score"],

                "category": information["category"],

                "spam_word": True

            })

        else:

            analysis_dictionary.append({

                "word": word,

                "risk": "Low",

                "score": 0,

                "category": "Normal",

                "spam_word": False

            })

    return analysis_dictionary

# ============================================================
# SPAM PHRASE ANALYSIS
# ============================================================

def analyse_spam_phrases(text):

    detected_phrases = []

    lowercase_text = text.lower()

    for phrase, information in SPAM_PHRASES.items():

        if phrase in lowercase_text:

            detected_phrases.append({

                "phrase": phrase,

                "risk": information["risk"],

                "score": information["score"],

                "category": information["category"]

            })

    return detected_phrases

# ============================================================
# RISK COUNT ANALYSIS
# ============================================================

def calculate_risk_distribution(analysis_dictionary):

    risk_count = {

        "High": 0,

        "Medium": 0,

        "Low": 0

    }

    for item in analysis_dictionary:

        risk = item["risk"]

        risk_count[risk] += 1

    total = len(analysis_dictionary)

    if total == 0:

        return {

            "High": 0,

            "Medium": 0,

            "Low": 0

        }

    percentages = {

        "High":
            (
                risk_count["High"]
                /
                total
            ) * 100,

        "Medium":
            (
                risk_count["Medium"]
                /
                total
            ) * 100,

        "Low":
            (
                risk_count["Low"]
                /
                total
            ) * 100

    }

    return percentages

# ============================================================
# SPAM SCORE CALCULATION
# ============================================================

def calculate_spam_score(
        analysis_dictionary,
        phrase_analysis,
        urls,
        emails,
        phone_numbers,
        currency_values,
        punctuation_matches,
        capital_ratio,
        repeated_words
):

    total_score = 0

    # --------------------------------------------------------
    # WORD SCORE
    # --------------------------------------------------------

    for item in analysis_dictionary:

        total_score += item["score"]

    # --------------------------------------------------------
    # PHRASE SCORE
    # --------------------------------------------------------

    for phrase in phrase_analysis:

        total_score += phrase["score"]

    # --------------------------------------------------------
    # URL SCORE
    # --------------------------------------------------------

    if len(urls) > 0:

        total_score += len(urls) * 5

    # --------------------------------------------------------
    # EMAIL SCORE
    # --------------------------------------------------------

    if len(emails) > 0:

        total_score += len(emails) * 2

    # --------------------------------------------------------
    # PHONE NUMBER SCORE
    # --------------------------------------------------------

    if len(phone_numbers) > 0:

        total_score += len(phone_numbers) * 3

    # --------------------------------------------------------
    # CURRENCY SCORE
    # --------------------------------------------------------

    if len(currency_values) > 0:

        total_score += len(currency_values) * 4

    # --------------------------------------------------------
    # EXCESSIVE PUNCTUATION SCORE
    # --------------------------------------------------------

    if len(punctuation_matches) > 0:

        total_score += len(punctuation_matches) * 5

    # --------------------------------------------------------
    # EXCESSIVE CAPITAL LETTER SCORE
    # --------------------------------------------------------

    if capital_ratio >= 70:

        total_score += 10

    elif capital_ratio >= 50:

        total_score += 6

    elif capital_ratio >= 30:

        total_score += 3

    # --------------------------------------------------------
    # REPEATED WORD SCORE
    # --------------------------------------------------------

    for word, count in repeated_words.items():

        total_score += (
            count - 2
        ) * 2

    return total_score

# ============================================================
# FINAL SPAM CLASSIFICATION
# ============================================================

def classify_spam(score):

    if score >= 60:

        return "HIGHLY LIKELY SPAM"

    elif score >= 35:

        return "SPAM"

    elif score >= 20:

        return "SUSPICIOUS"

    else:

        return "NOT SPAM"

# ============================================================
# SPAM CONFIDENCE
# ============================================================

def calculate_confidence(score):

    confidence = min(
        (score / 60) * 100,
        100
    )

    return confidence

# ============================================================
# COMPLETE STRUCTURED ANALYSIS
# ============================================================

def perform_complete_analysis(paragraph):

    # --------------------------------------------------------
    # BASIC PROCESSING
    # --------------------------------------------------------

    lowercase_paragraph = convert_to_lowercase(paragraph)

    sentences = tokenize_sentences(paragraph)

    all_tokens = tokenize_words(lowercase_paragraph)

    filtered_words = remove_unwanted_words(all_tokens)

    # --------------------------------------------------------
    # WORD ANALYSIS
    # --------------------------------------------------------

    word_analysis = analyse_words(
        filtered_words
    )

    # --------------------------------------------------------
    # PHRASE ANALYSIS
    # --------------------------------------------------------

    phrase_analysis = analyse_spam_phrases(
        paragraph
    )

    # --------------------------------------------------------
    # PATTERN ANALYSIS
    # --------------------------------------------------------

    urls = detect_urls(
        paragraph
    )

    emails = detect_emails(
        paragraph
    )

    phone_numbers = detect_phone_numbers(
        paragraph
    )

    currency_values = detect_currency(
        paragraph
    )

    punctuation_matches = detect_excessive_punctuation(
        paragraph
    )

    capital_ratio = calculate_capital_ratio(
        paragraph
    )

    repeated_words = detect_repeated_words(
        filtered_words
    )

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    risk_distribution = calculate_risk_distribution(
        word_analysis
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    spam_score = calculate_spam_score(

        word_analysis,

        phrase_analysis,

        urls,

        emails,

        phone_numbers,

        currency_values,

        punctuation_matches,

        capital_ratio,

        repeated_words

    )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    result = classify_spam(
        spam_score
    )

    confidence = calculate_confidence(
        spam_score
    )

    # --------------------------------------------------------
    # RETURN STRUCTURED RESULT
    # --------------------------------------------------------

    return {

        "original_text":
            paragraph,

        "lowercase_text":
            lowercase_paragraph,

        "sentences":
            sentences,

        "all_tokens":
            all_tokens,

        "filtered_words":
            filtered_words,

        "word_analysis":
            word_analysis,

        "phrase_analysis":
            phrase_analysis,

        "risk_distribution":
            risk_distribution,

        "detected_urls":
            urls,

        "detected_emails":
            emails,

        "detected_phone_numbers":
            phone_numbers,

        "detected_currency":
            currency_values,

        "excessive_punctuation":
            punctuation_matches,

        "capital_ratio":
            capital_ratio,

        "repeated_words":
            repeated_words,

        "spam_score":
            spam_score,

        "result":
            result,

        "confidence":
            confidence

    }

# ============================================================
# BAG OF WORDS
# ============================================================

def build_document_term_matrix(paragraph):

    sentences = sent_tokenize(
        paragraph
    )

    tokenized_sentences = []

    for sentence in sentences:

        lower = convert_to_lowercase(
            sentence
        )

        tokens = tokenize_words(
            lower
        )

        filtered = [

            word

            for word in tokens

            if word not in stop_words

            and not all(
                char in string.punctuation
                for char in word
            )

        ]

        tokenized_sentences.append(
            filtered
        )

    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    vocabulary = sorted(

        set(

            word

            for tokens in tokenized_sentences

            for word in tokens

        )

    )

    # --------------------------------------------------------
    # STANDARD BAG OF WORDS
    # --------------------------------------------------------

    document_vectors = []

    for tokens in tokenized_sentences:

        vector = [

            tokens.count(word)

            for word in vocabulary

        ]

        document_vectors.append(
            vector
        )

    return (

        sentences,

        tokenized_sentences,

        vocabulary,

        document_vectors

    )

# ============================================================
# BINARY BAG OF WORDS
# ============================================================

def build_binary_bow(paragraph):

    sentences = sent_tokenize(
        paragraph
    )

    vectorizer = CountVectorizer(

        binary=True,

        stop_words="english"

    )

    matrix = vectorizer.fit_transform(
        sentences
    )

    vocabulary = vectorizer.get_feature_names_out()

    binary_matrix = matrix.toarray()

    return (

        vocabulary,

        binary_matrix

    )

# ============================================================
# SCIKIT-LEARN BAG OF WORDS
# ============================================================

def sklearn_bag_of_words(paragraph):

    sentences = sent_tokenize(
        paragraph
    )

    vectorizer = CountVectorizer(

        stop_words="english"

    )

    matrix = vectorizer.fit_transform(
        sentences
    )

    vocabulary = vectorizer.get_feature_names_out()

    bow_matrix = matrix.toarray()

    return (

        vocabulary,

        bow_matrix

    )

# ============================================================
# TF-IDF ANALYSIS
# ============================================================

def build_tfidf_matrix(paragraph):

    sentences = sent_tokenize(
        paragraph
    )

    if len(sentences) == 0:

        return [], np.array([])

    vectorizer = TfidfVectorizer(

        stop_words="english"

    )

    matrix = vectorizer.fit_transform(
        sentences
    )

    vocabulary = vectorizer.get_feature_names_out()

    tfidf_matrix = matrix.toarray()

    return (

        vocabulary,

        tfidf_matrix

    )

# ============================================================
# N-GRAM ANALYSIS
# ============================================================

def generate_ngrams(paragraph, ngram_range=(1, 2)):

    sentences = sent_tokenize(
        paragraph
    )

    if len(sentences) == 0:

        return [], np.array([])

    vectorizer = CountVectorizer(

        stop_words="english",

        ngram_range=ngram_range

    )

    matrix = vectorizer.fit_transform(
        sentences
    )

    vocabulary = vectorizer.get_feature_names_out()

    ngram_matrix = matrix.toarray()

    return (

        vocabulary,

        ngram_matrix

    )

# ============================================================
# CATEGORY ANALYSIS
# ============================================================

def analyse_categories(word_analysis):

    categories = defaultdict(int)

    for item in word_analysis:

        if item["spam_word"]:

            category = item["category"]

            categories[category] += 1

    return dict(categories)

# ============================================================
# STRUCTURED CONSOLE OUTPUT
# ============================================================

def print_analysis_results(results):

    print("\n")

    print("=" * 80)

    print(
        "ADVANCED NLP SPAM DETECTION RESULTS"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # ORIGINAL TEXT
    # --------------------------------------------------------

    print("\n[1] ORIGINAL TEXT")

    print("-" * 80)

    print(
        results["original_text"]
    )

    # --------------------------------------------------------
    # LOWERCASE
    # --------------------------------------------------------

    print("\n[2] LOWERCASE TEXT")

    print("-" * 80)

    print(
        results["lowercase_text"]
    )

    # --------------------------------------------------------
    # SENTENCES
    # --------------------------------------------------------

    print("\n[3] SENTENCE TOKENIZATION")

    print("-" * 80)

    for index, sentence in enumerate(
        results["sentences"],
        start=1
    ):

        print(

            f"S{index}: {sentence}"

        )

    # --------------------------------------------------------
    # TOKENS
    # --------------------------------------------------------

    print("\n[4] ALL TOKENS")

    print("-" * 80)

    print(
        results["all_tokens"]
    )

    # --------------------------------------------------------
    # FILTERED WORDS
    # --------------------------------------------------------

    print("\n[5] FILTERED WORDS")

    print("-" * 80)

    print(
        results["filtered_words"]
    )

    # --------------------------------------------------------
    # WORD ANALYSIS
    # --------------------------------------------------------

    print("\n[6] WORD RISK ANALYSIS")

    print("-" * 80)

    print(

        f"{'WORD':<20}"

        f"{'RISK':<12}"

        f"{'SCORE':<10}"

        f"{'CATEGORY':<25}"

        f"{'SPAM WORD'}"

    )

    print("-" * 80)

    for item in results["word_analysis"]:

        print(

            f"{item['word']:<20}"

            f"{item['risk']:<12}"

            f"{item['score']:<10}"

            f"{item['category']:<25}"

            f"{str(item['spam_word'])}"

        )

    # --------------------------------------------------------
    # PHRASES
    # --------------------------------------------------------

    print("\n[7] DETECTED SPAM PHRASES")

    print("-" * 80)

    if results["phrase_analysis"]:

        for phrase in results["phrase_analysis"]:

            print(

                f"Phrase   : {phrase['phrase']}"

            )

            print(

                f"Risk     : {phrase['risk']}"

            )

            print(

                f"Score    : {phrase['score']}"

            )

            print(

                f"Category : {phrase['category']}"

            )

            print()

    else:

        print(
            "No suspicious spam phrases detected."
        )

    # --------------------------------------------------------
    # SPECIAL PATTERNS
    # --------------------------------------------------------

    print("\n[8] SPECIAL PATTERN ANALYSIS")

    print("-" * 80)

    print(
        f"URLs Found: {results['detected_urls']}"
    )

    print(
        f"Emails Found: {results['detected_emails']}"
    )

    print(
        f"Phone Numbers Found: "
        f"{results['detected_phone_numbers']}"
    )

    print(
        f"Currency Values Found: "
        f"{results['detected_currency']}"
    )

    print(
        f"Excessive Punctuation: "
        f"{results['excessive_punctuation']}"
    )

    print(
        f"Capital Letter Ratio: "
        f"{results['capital_ratio']:.2f}%"
    )

    print(
        f"Repeated Words: "
        f"{results['repeated_words']}"
    )

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    print("\n[9] RISK DISTRIBUTION")

    print("-" * 80)

    for risk, percentage in (
        results["risk_distribution"].items()
    ):

        print(

            f"{risk:<10}: "
            f"{percentage:.2f}%"

        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n")

    print("=" * 80)

    print(
        "FINAL SPAM CLASSIFICATION"
    )

    print("=" * 80)

    print(

        f"\nSpam Score : "
        f"{results['spam_score']}"

    )

    print(

        f"Confidence : "
        f"{results['confidence']:.2f}%"

    )

    print(

        f"\nVERDICT: "
        f">>> {results['result']} <<<"

    )

# ============================================================
# DOCUMENT TERM MATRIX OUTPUT
# ============================================================

def print_document_term_matrix(

        sentences,

        tokenized_sentences,

        vocabulary,

        document_vectors

):

    print("\n")

    print("=" * 80)

    print(
        "STANDARD BAG OF WORDS"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # SENTENCES
    # --------------------------------------------------------

    print("\nSentences:")

    for index, sentence in enumerate(
        sentences,
        start=1
    ):

        print(
            f"S{index}: {sentence}"
        )

    # --------------------------------------------------------
    # TOKENIZED SENTENCES
    # --------------------------------------------------------

    print("\nProcessed Sentences:")

    for index, tokens in enumerate(
        tokenized_sentences,
        start=1
    ):

        print(
            f"S{index}: {tokens}"
        )

    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    print("\nVocabulary:")

    print(
        vocabulary
    )

    # --------------------------------------------------------
    # MATRIX
    # --------------------------------------------------------

    print(
        "\nDocument-Term Matrix:"
    )

    if len(vocabulary) == 0:

        print(
            "No vocabulary generated."
        )

        return

    header = (

        f"{'Sentence':<12}"

        +

        "".join(

            f"{word:<15}"

            for word in vocabulary

        )

    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for index, vector in enumerate(
        document_vectors,
        start=1
    ):

        row = (

            f"S{index:<11}"

            +

            "".join(

                f"{count:<15}"

                for count in vector

            )

        )

        print(
            row
        )

    # --------------------------------------------------------
    # NUMERICAL VECTORS
    # --------------------------------------------------------

    print(
        "\nNumerical Vectors:"
    )

    for index, vector in enumerate(
        document_vectors,
        start=1
    ):

        print(

            f"S{index}: "
            f"{vector}"

        )

# ============================================================
# BINARY BAG OF WORDS OUTPUT
# ============================================================

def print_binary_bow(

        vocabulary,

        matrix

):

    print("\n")

    print("=" * 80)

    print(
        "BINARY BAG OF WORDS"
    )

    print("=" * 80)

    print("\nVocabulary:")

    print(
        list(vocabulary)
    )

    print(
        "\nBinary Matrix:"
    )

    for index, vector in enumerate(
        matrix,
        start=1
    ):

        print(

            f"S{index}: "
            f"{vector.tolist()}"

        )

# ============================================================
# TF-IDF OUTPUT
# ============================================================

def print_tfidf(

        vocabulary,

        matrix

):

    print("\n")

    print("=" * 80)

    print(
        "TF-IDF ANALYSIS"
    )

    print("=" * 80)

    print("\nVocabulary:")

    print(
        list(vocabulary)
    )

    print(
        "\nTF-IDF Matrix:"
    )

    for index, vector in enumerate(
        matrix,
        start=1
    ):

        formatted_vector = [

            round(
                value,
                4
            )

            for value in vector

        ]

        print(

            f"S{index}: "
            f"{formatted_vector}"

        )

# ============================================================
# N-GRAM OUTPUT
# ============================================================

def print_ngrams(

        vocabulary,

        matrix

):

    print("\n")

    print("=" * 80)

    print(
        "N-GRAM ANALYSIS"
    )

    print("=" * 80)

    print("\nN-Gram Vocabulary:")

    print(
        list(vocabulary)
    )

    print(
        "\nN-Gram Matrix:"
    )

    for index, vector in enumerate(
        matrix,
        start=1
    ):

        print(

            f"S{index}: "
            f"{vector.tolist()}"

        )

# ============================================================
# VISUALIZATION DASHBOARD
# ============================================================

def plot_all(

        results,

        vocabulary,

        document_vectors

):

    risk_colors = {

        "High":
            "#e74c3c",

        "Medium":
            "#f39c12",

        "Low":
            "#2ecc71"

    }

    fig = plt.figure(
        figsize=(20, 24)
    )

    fig.patch.set_facecolor(
        "#1e1e2e"
    )

    fig.suptitle(

        "Advanced NLP Spam Detection Dashboard",

        fontsize=24,

        fontweight="bold",

        color="white",

        y=0.98

    )

    gs = fig.add_gridspec(

        4,

        2,

        hspace=0.55,

        wspace=0.4,

        left=0.07,

        right=0.97,

        top=0.94,

        bottom=0.05

    )

    # ========================================================
    # GRAPH 1
    # WORD RISK ANALYSIS
    # ========================================================

    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    spam_items = [

        item

        for item in results["word_analysis"]

        if item["spam_word"]

    ]

    if spam_items:

        words = [

            item["word"]

            for item in spam_items

        ]

        scores = [

            item["score"]

            for item in spam_items

        ]

        colors = [

            risk_colors[
                item["risk"]
            ]

            for item in spam_items

        ]

        ax1.barh(

            words,

            scores,

            color=colors,

            edgecolor="white"

        )

    ax1.set_facecolor(
        "#2a2a3e"
    )

    ax1.set_title(

        "Spam Word Risk Score",

        color="white",

        fontweight="bold"

    )

    ax1.tick_params(
        colors="white"
    )

    # ========================================================
    # GRAPH 2
    # RISK DISTRIBUTION
    # ========================================================

    ax2 = fig.add_subplot(
        gs[0, 1]
    )

    labels = [

        "High",

        "Medium",

        "Low"

    ]

    values = [

        results[
            "risk_distribution"
        ][label]

        for label in labels

    ]

    colors = [

        risk_colors[label]

        for label in labels

    ]

    ax2.bar(

        labels,

        values,

        color=colors,

        edgecolor="white"

    )

    ax2.set_facecolor(
        "#2a2a3e"
    )

    ax2.set_title(

        "Risk Distribution (%)",

        color="white",

        fontweight="bold"

    )

    ax2.tick_params(
        colors="white"
    )

    # ========================================================
    # GRAPH 3
    # WORD FREQUENCY
    # ========================================================

    ax3 = fig.add_subplot(
        gs[1, 0]
    )

    frequency = Counter(
        results["filtered_words"]
    )

    words = list(
        frequency.keys()
    )

    counts = list(
        frequency.values()
    )

    ax3.bar(

        words,

        counts,

        edgecolor="white"

    )

    ax3.set_facecolor(
        "#2a2a3e"
    )

    ax3.set_title(

        "Filtered Word Frequency",

        color="white",

        fontweight="bold"

    )

    ax3.tick_params(
        colors="white"
    )

    plt.setp(

        ax3.get_xticklabels(),

        rotation=45,

        ha="right"

    )

    # ========================================================
    # GRAPH 4
    # SPAM SCORE
    # ========================================================

    ax4 = fig.add_subplot(
        gs[1, 1]
    )

    score = results[
        "spam_score"
    ]

    confidence = results[
        "confidence"
    ]

    ax4.bar(

        ["Spam Score"],

        [score],

        edgecolor="white"

    )

    ax4.axhline(

        20,

        linestyle="--",

        color="#f39c12",

        label="Suspicious"

    )

    ax4.axhline(

        35,

        linestyle="--",

        color="#e67e22",

        label="Spam"

    )

    ax4.axhline(

        60,

        linestyle="--",

        color="#e74c3c",

        label="Highly Likely Spam"

    )

    ax4.set_facecolor(
        "#2a2a3e"
    )

    ax4.set_title(

        f"Spam Score | Confidence: {confidence:.1f}%",

        color="white",

        fontweight="bold"

    )

    ax4.tick_params(
        colors="white"
    )

    ax4.legend(

        facecolor="#2a2a3e",

        labelcolor="white"

    )

    # ========================================================
    # GRAPH 5
    # DOCUMENT TERM MATRIX
    # ========================================================

    ax5 = fig.add_subplot(
        gs[2, :]
    )

    matrix = np.array(
        document_vectors
    )

    if matrix.size > 0:

        im = ax5.imshow(

            matrix,

            aspect="auto",

            cmap="YlOrRd",

            interpolation="nearest"

        )

        ax5.set_facecolor(
            "#2a2a3e"
        )

        ax5.set_title(

            "Document-Term Matrix",

            color="white",

            fontweight="bold"

        )

        ax5.set_xticks(
            range(
                len(vocabulary)
            )
        )

        ax5.set_xticklabels(

            vocabulary,

            rotation=45,

            ha="right",

            color="white"

        )

        ax5.set_yticks(

            range(
                len(document_vectors)
            )
        )

        ax5.set_yticklabels(

            [

                f"S{i + 1}"

                for i in range(
                    len(document_vectors)
                )

            ],

            color="white"

        )

        for i in range(
            matrix.shape[0]
        ):

            for j in range(
                matrix.shape[1]
            ):

                ax5.text(

                    j,

                    i,

                    str(
                        matrix[i, j]
                    ),

                    ha="center",

                    va="center",

                    color="black"

                    if matrix[i, j]
                    >
                    matrix.max() / 2

                    else "white"

                )

        cbar = fig.colorbar(

            im,

            ax=ax5

        )

        cbar.set_label(
            "Word Frequency"
        )

    # ========================================================
    # GRAPH 6
    # CAPITAL / PATTERN ANALYSIS
    # ========================================================

    ax6 = fig.add_subplot(
        gs[3, :]
    )

    pattern_labels = [

        "URLs",

        "Emails",

        "Phones",

        "Currency",

        "Punctuation",

        "Repeated Words",

        "Capital Ratio"

    ]

    pattern_values = [

        len(
            results[
                "detected_urls"
            ]
        ),

        len(
            results[
                "detected_emails"
            ]
        ),

        len(
            results[
                "detected_phone_numbers"
            ]
        ),

        len(
            results[
                "detected_currency"
            ]
        ),

        len(
            results[
                "excessive_punctuation"
            ]
        ),

        len(
            results[
                "repeated_words"
            ]
        ),

        results[
            "capital_ratio"
        ]

    ]

    ax6.bar(

        pattern_labels,

        pattern_values,

        edgecolor="white"

    )

    ax6.set_facecolor(
        "#2a2a3e"
    )

    ax6.set_title(

        "Special Pattern Analysis",

        color="white",

        fontweight="bold"

    )

    ax6.tick_params(
        colors="white"
    )

    plt.setp(

        ax6.get_xticklabels(),

        rotation=30,

        ha="right"

    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.savefig(

        "advanced_nlp_analysis.png",

        dpi=150,

        bbox_inches="tight",

        facecolor=fig.get_facecolor()

    )

    print(

        "\n[Visualization saved as advanced_nlp_analysis.png]"

    )

    plt.show()

# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")

    print("=" * 80)

    print(
        "ADVANCED NLP SPAM DETECTION SYSTEM"
    )

    print("=" * 80)

    print(

        "\nEnter a paragraph."

    )

    print(

        "Press ENTER twice when finished."

    )

    lines = []

    while True:

        line = input()

        if line == "":

            break

        lines.append(
            line
        )

    paragraph = " ".join(
        lines
    )

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not paragraph.strip():

        print(
            "\nNo text entered."
        )

        return

    # --------------------------------------------------------
    # COMPLETE ANALYSIS
    # --------------------------------------------------------

    results = perform_complete_analysis(
        paragraph
    )

    # --------------------------------------------------------
    # STRUCTURED OUTPUT
    # --------------------------------------------------------

    print_analysis_results(
        results
    )

    # --------------------------------------------------------
    # STANDARD BAG OF WORDS
    # --------------------------------------------------------

    (
        sentences,

        tokenized_sentences,

        vocabulary,

        document_vectors

    ) = build_document_term_matrix(
        paragraph
    )

    print_document_term_matrix(

        sentences,

        tokenized_sentences,

        vocabulary,

        document_vectors

    )

    # --------------------------------------------------------
    # BINARY BAG OF WORDS
    # --------------------------------------------------------

    try:

        (

            binary_vocabulary,

            binary_matrix

        ) = build_binary_bow(
            paragraph
        )

        print_binary_bow(

            binary_vocabulary,

            binary_matrix

        )

    except ValueError:

        print(

            "\nBinary Bag of Words could not be generated."

        )

    # --------------------------------------------------------
    # SCIKIT-LEARN BAG OF WORDS
    # --------------------------------------------------------

    try:

        (

            sklearn_vocabulary,

            sklearn_matrix

        ) = sklearn_bag_of_words(
            paragraph
        )

        print("\n")

        print("=" * 80)

        print(
            "SCIKIT-LEARN BAG OF WORDS"
        )

        print("=" * 80)

        print(
            "\nVocabulary:"
        )

        print(
            list(
                sklearn_vocabulary
            )
        )

        print(
            "\nMatrix:"
        )

        for index, vector in enumerate(

            sklearn_matrix,

            start=1

        ):

            print(

                f"S{index}: "
                f"{vector.tolist()}"

            )

    except ValueError:

        print(

            "\nScikit-Learn Bag of Words could not be generated."

        )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    try:

        (

            tfidf_vocabulary,

            tfidf_matrix

        ) = build_tfidf_matrix(
            paragraph
        )

        print_tfidf(

            tfidf_vocabulary,

            tfidf_matrix

        )

    except ValueError:

        print(

            "\nTF-IDF could not be generated."

        )

    # --------------------------------------------------------
    # N-GRAMS
    # --------------------------------------------------------

    try:

        (

            ngram_vocabulary,

            ngram_matrix

        ) = generate_ngrams(

            paragraph,

            ngram_range=(1, 2)

        )

        print_ngrams(

            ngram_vocabulary,

            ngram_matrix

        )

    except ValueError:

        print(

            "\nN-Gram analysis could not be generated."

        )

    # --------------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------------

    category_analysis = analyse_categories(

        results[
            "word_analysis"
        ]

    )

    print("\n")

    print("=" * 80)

    print(
        "SPAM CATEGORY ANALYSIS"
    )

    print("=" * 80)

    if category_analysis:

        for category, count in (
            category_analysis.items()
        ):

            print(

                f"{category:<30} : "
                f"{count}"

            )

    else:

        print(
            "No spam categories detected."
        )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    print("\n")

    print("=" * 80)

    print(
        "GENERATING VISUALIZATION DASHBOARD"
    )

    print("=" * 80)

    plot_all(

        results,

        vocabulary,

        document_vectors

    )

# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
