# ============================================================
# ROBUST TOP-DOWN PARSER FOR ENGLISH
# ============================================================
#
# NLP / COMPILER DESIGN DEMONSTRATION
#
# Features:
#   - Tokenization
#   - NLTK POS tagging
#   - WordNet lexical fallback
#   - Grammar-based lexical classification
#   - Recursive-descent TOP-DOWN parsing
#   - Backtracking
#   - Complete parse tree
#   - Grammar derivation display
#   - Detailed token analysis
#   - Error reporting
#
# ------------------------------------------------------------
# MAIN GRAMMAR
# ------------------------------------------------------------
#
# S
#   -> CLAUSE
#
# CLAUSE
#   -> NP VP
#   -> NP VP CONJ CLAUSE
#
# NP
#   -> PRON
#   -> PROPN
#   -> DET NOMINAL
#   -> NOMINAL
#   -> NUM NOMINAL
#
# NOMINAL
#   -> ADJ_PHRASE NBAR
#   -> NBAR
#
# NBAR
#   -> N
#   -> N NBAR
#   -> N PP
#   -> N NBAR PP
#
# ADJ_PHRASE
#   -> ADJ
#   -> ADV ADJ
#   -> ADJ ADJ_PHRASE
#
# VP
#   -> V
#   -> AUX V
#   -> AUX VP
#   -> V NP
#   -> V PP
#   -> V NP PP
#   -> V ADJ_PHRASE
#   -> AUX V NP
#   -> AUX V PP
#   -> AUX V NP PP
#
# PP
#   -> PREP NP
#
# ------------------------------------------------------------
#
# Example:
#
#   The intelligent student solved the difficult problem
#
# produces:
#
# S
# └── CLAUSE
#     ├── NP
#     │   ├── DET
#     │   │   └── The
#     │   └── NOMINAL
#     │       ├── ADJ_PHRASE
#     │       │   └── ADJ
#     │       │       └── intelligent
#     │       └── NBAR
#     │           └── N
#     │               └── student
#     │
#     └── VP
#         ├── V
#         │   └── solved
#         └── NP
#             ├── DET
#             │   └── the
#             └── NOMINAL
#                 ├── ADJ_PHRASE
#                 │   └── ADJ
#                 │       └── difficult
#                 └── NBAR
#                     └── N
#                         └── problem
#
# ============================================================


import sys
import subprocess
import importlib


# ============================================================
# 1. MAKE SURE NLTK IS INSTALLED
# ============================================================

def ensure_nltk():

    try:
        importlib.import_module("nltk")

    except ImportError:

        print("NLTK is not installed.")
        print("Installing NLTK...")

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "nltk"
            ]
        )


ensure_nltk()


# ============================================================
# 2. IMPORT NLTK
# ============================================================

import nltk

from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet


# ============================================================
# 3. DOWNLOAD REQUIRED NLTK RESOURCES
# ============================================================

def ensure_nltk_resources():

    resources = [
        (
            "tokenizers/punkt",
            "punkt"
        ),
        (
            "tokenizers/punkt_tab",
            "punkt_tab"
        ),
        (
            "taggers/averaged_perceptron_tagger",
            "averaged_perceptron_tagger"
        ),
        (
            "taggers/averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger_eng"
        ),
        (
            "corpora/wordnet",
            "wordnet"
        ),
        (
            "corpora/omw-1.4",
            "omw-1.4"
        )
    ]

    for path, package in resources:

        try:

            nltk.data.find(path)

        except LookupError:

            print(
                f"Downloading NLTK resource: {package}"
            )

            nltk.download(package)


ensure_nltk_resources()


# ============================================================
# 4. TREE NODE
# ============================================================

class Node:

    def __init__(
        self,
        label,
        children=None,
        terminal=False
    ):

        self.label = label

        self.children = (
            children
            if children is not None
            else []
        )

        self.terminal = terminal

    def add_child(self, child):

        self.children.append(child)


# ============================================================
# 5. LEXICAL ANALYZER
# ============================================================

class LexicalAnalyzer:

    """
    Converts NLTK POS tags into grammar categories.

    We deliberately do NOT maintain a giant list such as:

        nouns = {"dog", "cat", "boy", ...}

    because that defeats the purpose of making the parser
    general.

    NLTK POS tagging is the primary source.

    WordNet is used as a fallback.
    """

    def __init__(self):

        # ----------------------------------------------------
        # FUNCTION WORDS
        # ----------------------------------------------------

        self.determiners = {
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "some",
            "any",
            "each",
            "every",
            "either",
            "neither",
            "enough",
            "several",
            "many",
            "few",
            "much",
            "little"
        }

        self.prepositions = {
            "about",
            "above",
            "across",
            "after",
            "against",
            "along",
            "among",
            "around",
            "at",
            "before",
            "behind",
            "below",
            "beneath",
            "beside",
            "between",
            "beyond",
            "by",
            "despite",
            "down",
            "during",
            "except",
            "for",
            "from",
            "in",
            "inside",
            "into",
            "near",
            "of",
            "off",
            "on",
            "onto",
            "over",
            "past",
            "through",
            "throughout",
            "to",
            "toward",
            "under",
            "underneath",
            "until",
            "up",
            "upon",
            "with",
            "within",
            "without"
        }

        self.conjunctions = {
            "and",
            "or",
            "but",
            "nor",
            "yet",
            "so"
        }

        # ----------------------------------------------------
        # AUXILIARY VERBS
        # ----------------------------------------------------

        self.auxiliaries = {
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "may",
            "might",
            "must",
            "shall",
            "should",
            "will",
            "would"
        }

        # ----------------------------------------------------
        # COMMON COPULAR FORMS
        # ----------------------------------------------------

        self.copulas = {
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being"
        }

        # ----------------------------------------------------
        # POS TAGS WHICH REPRESENT AUXILIARIES
        # ----------------------------------------------------

        self.auxiliary_tags = {
            "MD"
        }

    # ========================================================
    # WORDNET HELPERS
    # ========================================================

    def has_noun(self, word):

        return bool(
            wordnet.synsets(
                word,
                pos=wordnet.NOUN
            )
        )

    def has_verb(self, word):

        return bool(
            wordnet.synsets(
                word,
                pos=wordnet.VERB
            )
        )

    def has_adjective(self, word):

        return bool(
            wordnet.synsets(
                word,
                pos=wordnet.ADJ
            )
        )

    def has_adverb(self, word):

        return bool(
            wordnet.synsets(
                word,
                pos=wordnet.ADV
            )
        )

    # ========================================================
    # CLASSIFY
    # ========================================================

    def classify(self, word, pos):

        lower = word.lower()

        # ----------------------------------------------------
        # PUNCTUATION
        # ----------------------------------------------------

        if word in {
            ".",
            ",",
            "!",
            "?",
            ";",
            ":"
        }:

            return "PUNCT"

        # ----------------------------------------------------
        # CONJUNCTION
        # ----------------------------------------------------

        if lower in self.conjunctions:

            return "CONJ"

        # ----------------------------------------------------
        # DETERMINER
        # ----------------------------------------------------

        if lower in self.determiners:

            return "DET"

        # ----------------------------------------------------
        # PREPOSITION
        # ----------------------------------------------------

        if lower in self.prepositions:

            return "PREP"

        # ----------------------------------------------------
        # AUXILIARY
        # ----------------------------------------------------

        if lower in self.auxiliaries:

            return "AUX"

        if pos in self.auxiliary_tags:

            return "AUX"

        # ----------------------------------------------------
        # PRONOUN
        # ----------------------------------------------------

        if pos in {
            "PRP",
            "PRP$",
            "WP",
            "WP$"
        }:

            return "PRON"

        # ----------------------------------------------------
        # NUMBERS
        # ----------------------------------------------------

        if pos == "CD":

            return "NUM"

        # ----------------------------------------------------
        # ADJECTIVE
        # ----------------------------------------------------

        if pos.startswith("JJ"):

            return "ADJ"

        # ----------------------------------------------------
        # ADVERB
        # ----------------------------------------------------

        if pos.startswith("RB"):

            return "ADV"

        # ----------------------------------------------------
        # VERB
        # ----------------------------------------------------

        if pos.startswith("VB"):

            # Auxiliary forms were handled above.

            if lower not in self.auxiliaries:

                return "V"

        # ----------------------------------------------------
        # PROPER NOUN
        # ----------------------------------------------------

        if pos in {
            "NNP",
            "NNPS"
        }:

            return "PROPN"

        # ----------------------------------------------------
        # NORMAL NOUN
        # ----------------------------------------------------

        if pos.startswith("NN"):

            return "N"

        # ----------------------------------------------------
        # WORDNET FALLBACK
        # ----------------------------------------------------

        if self.has_noun(lower):

            return "N"

        if self.has_verb(lower):

            return "V"

        if self.has_adjective(lower):

            return "ADJ"

        if self.has_adverb(lower):

            return "ADV"

        return "UNKNOWN"


# ============================================================
# 6. TOP-DOWN PARSER
# ============================================================

class TopDownParser:

    def __init__(
        self,
        tokens,
        tagged_tokens,
        verbose=True
    ):

        self.tokens = tokens

        self.tagged_tokens = tagged_tokens

        self.position = 0

        self.verbose = verbose

        self.lexer = LexicalAnalyzer()

        self.categories = []

        # ----------------------------------------------------
        # Classify every token
        # ----------------------------------------------------

        for word, pos in tagged_tokens:

            category = self.lexer.classify(
                word,
                pos
            )

            self.categories.append(category)

        # ----------------------------------------------------
        # Derivation history
        # ----------------------------------------------------

        self.derivation = []

        # ----------------------------------------------------
        # Error tracking
        # ----------------------------------------------------

        self.farthest_position = 0

        self.farthest_expected = set()

    # ========================================================
    # DEBUG
    # ========================================================

    def debug(self, message):

        if self.verbose:

            print(message)

    # ========================================================
    # SAVE PARSER POSITION
    # ========================================================

    def save(self):

        return self.position

    # ========================================================
    # RESTORE PARSER POSITION
    # ========================================================

    def restore(self, position):

        self.position = position

    # ========================================================
    # RECORD FAILURE
    # ========================================================

    def record_failure(self, expected):

        if self.position > self.farthest_position:

            self.farthest_position = self.position

            self.farthest_expected = {
                expected
            }

        elif self.position == self.farthest_position:

            self.farthest_expected.add(
                expected
            )

    # ========================================================
    # CURRENT TOKEN
    # ========================================================

    def current_token(self):

        if self.position < len(self.tokens):

            return self.tokens[
                self.position
            ]

        return "<END OF INPUT>"

    # ========================================================
    # CURRENT CATEGORY
    # ========================================================

    def current_category(self):

        if self.position < len(
            self.categories
        ):

            return self.categories[
                self.position
            ]

        return "EOF"

    # ========================================================
    # MATCH CATEGORY
    # ========================================================

    def match(self, category):

        return (
            self.current_category()
            == category
        )

    # ========================================================
    # CONSUME TERMINAL
    # ========================================================

    def consume(self, category):

        if self.match(category):

            word = self.current_token()

            self.debug(
                f"      MATCH {category} -> {word}"
            )

            self.position += 1

            return Node(
                category,
                [
                    Node(
                        word,
                        terminal=True
                    )
                ]
            )

        self.record_failure(category)

        return None

    # ========================================================
    # S
    #
    # S -> CLAUSE
    # ========================================================

    def parse_S(self):

        start = self.save()

        self.debug(
            "\n[S] S -> CLAUSE"
        )

        clause = self.parse_CLAUSE()

        if clause is not None:

            return Node(
                "S",
                [clause]
            )

        self.restore(start)

        return None

    # ========================================================
    # CLAUSE
    #
    # CLAUSE
    #   -> NP VP
    #   -> NP VP CONJ CLAUSE
    # ========================================================

    def parse_CLAUSE(self):

        start = self.save()

        self.debug(
            "[CLAUSE] Trying NP VP"
        )

        np = self.parse_NP()

        if np is None:

            self.restore(start)

            return None

        vp = self.parse_VP()

        if vp is None:

            self.restore(start)

            return None

        children = [
            np,
            vp
        ]

        # ----------------------------------------------------
        # Compound sentence
        # ----------------------------------------------------

        if self.match("CONJ"):

            conj = self.consume("CONJ")

            if conj is not None:

                clause = self.parse_CLAUSE()

                if clause is not None:

                    children.append(conj)

                    children.append(clause)

        return Node(
            "CLAUSE",
            children
        )

    # ========================================================
    # NP
    #
    # NP -> PRON
    # NP -> PROPN
    # NP -> DET NOMINAL
    # NP -> NOMINAL
    # NP -> NUM NOMINAL
    # ========================================================

    def parse_NP(self):

        start = self.save()

        self.debug(
            "  [NP] Trying NP"
        )

        # ----------------------------------------------------
        # PRONOUN
        # ----------------------------------------------------

        if self.match("PRON"):

            pron = self.consume("PRON")

            return Node(
                "NP",
                [pron]
            )

        # ----------------------------------------------------
        # PROPER NOUN
        # ----------------------------------------------------

        if self.match("PROPN"):

            propn = self.consume(
                "PROPN"
            )

            return Node(
                "NP",
                [propn]
            )

        # ----------------------------------------------------
        # NUMBER + NOMINAL
        # ----------------------------------------------------

        if self.match("NUM"):

            num = self.consume("NUM")

            nominal = self.parse_NOMINAL()

            if nominal is not None:

                return Node(
                    "NP",
                    [
                        num,
                        nominal
                    ]
                )

            self.restore(start)

        # ----------------------------------------------------
        # DETERMINER + NOMINAL
        # ----------------------------------------------------

        if self.match("DET"):

            det = self.consume("DET")

            nominal = self.parse_NOMINAL()

            if nominal is not None:

                return Node(
                    "NP",
                    [
                        det,
                        nominal
                    ]
                )

            self.restore(start)

        # ----------------------------------------------------
        # BARE NOMINAL
        # ----------------------------------------------------

        nominal = self.parse_NOMINAL()

        if nominal is not None:

            return Node(
                "NP",
                [nominal]
            )

        self.restore(start)

        return None

    # ========================================================
    # NOMINAL
    #
    # NOMINAL -> ADJ_PHRASE NBAR
    # NOMINAL -> NBAR
    # ========================================================

    def parse_NOMINAL(self):

        start = self.save()

        # ----------------------------------------------------
        # ADJECTIVE PHRASE + NBAR
        # ----------------------------------------------------

        adj_phrase = self.parse_ADJ_PHRASE()

        if adj_phrase is not None:

            nbar = self.parse_NBAR()

            if nbar is not None:

                return Node(
                    "NOMINAL",
                    [
                        adj_phrase,
                        nbar
                    ]
                )

            self.restore(start)

        # ----------------------------------------------------
        # NBAR
        # ----------------------------------------------------

        nbar = self.parse_NBAR()

        if nbar is not None:

            return Node(
                "NOMINAL",
                [nbar]
            )

        self.restore(start)

        return None

    # ========================================================
    # ADJECTIVE PHRASE
    #
    # ADJ_PHRASE
    #   -> ADJ
    #   -> ADV ADJ
    #   -> ADJ ADJ_PHRASE
    # ========================================================

    def parse_ADJ_PHRASE(self):

        start = self.save()

        # ----------------------------------------------------
        # ADV + ADJ
        # ----------------------------------------------------

        if self.match("ADV"):

            adv = self.consume("ADV")

            adj = self.consume("ADJ")

            if adj is not None:

                return Node(
                    "ADJ_PHRASE",
                    [
                        adv,
                        adj
                    ]
                )

            self.restore(start)

        # ----------------------------------------------------
        # ADJ
        # ----------------------------------------------------

        if self.match("ADJ"):

            adj = self.consume("ADJ")

            children = [adj]

            # ------------------------------------------------
            # Multiple adjectives
            # ------------------------------------------------

            while self.match("ADJ"):

                next_adj = self.consume(
                    "ADJ"
                )

                if next_adj is not None:

                    children.append(
                        next_adj
                    )

            return Node(
                "ADJ_PHRASE",
                children
            )

        self.restore(start)

        return None

    # ========================================================
    # NBAR
    #
    # NBAR -> N
    # NBAR -> N NBAR
    # NBAR -> N PP
    # ========================================================

    def parse_NBAR(self):

        start = self.save()

        # ----------------------------------------------------
        # NOUN
        # ----------------------------------------------------

        if self.match("N"):

            noun = self.consume("N")

            children = [noun]

            # ------------------------------------------------
            # Noun modifiers / PPs
            # ------------------------------------------------

            while True:

                pp_start = self.save()

                if self.match("PREP"):

                    pp = self.parse_PP()

                    if pp is not None:

                        children.append(pp)

                        continue

                self.restore(pp_start)

                break

            return Node(
                "NBAR",
                children
            )

        # ----------------------------------------------------
        # PROPER NOUN fallback
        # ----------------------------------------------------

        if self.match("PROPN"):

            propn = self.consume(
                "PROPN"
            )

            return Node(
                "NBAR",
                [propn]
            )

        self.restore(start)

        return None

    # ========================================================
    # VP
    #
    # VP -> V
    # VP -> V NP
    # VP -> V PP
    # VP -> V NP PP
    #
    # VP -> AUX V
    # VP -> AUX V NP
    # VP -> AUX V PP
    # VP -> AUX V NP PP
    #
    # VP -> AUX ADJ_PHRASE
    #
    # ========================================================

    def parse_VP(self):

        start = self.save()

        self.debug(
            "  [VP] Trying VP"
        )

        children = []

        # ----------------------------------------------------
        # AUXILIARY
        # ----------------------------------------------------

        if self.match("AUX"):

            aux = self.consume("AUX")

            children.append(aux)

            # ------------------------------------------------
            # Auxiliary + adjective
            #
            # Example:
            # The student is intelligent
            # ------------------------------------------------

            adj_phrase = self.parse_ADJ_PHRASE()

            if adj_phrase is not None:

                children.append(
                    adj_phrase
                )

                return Node(
                    "VP",
                    children
                )

            # ------------------------------------------------
            # Auxiliary + verb
            # ------------------------------------------------

            if not self.match("V"):

                self.restore(start)

                return None

            verb = self.consume("V")

            children.append(verb)

        else:

            # ------------------------------------------------
            # Direct verb
            # ------------------------------------------------

            if not self.match("V"):

                self.restore(start)

                return None

            verb = self.consume("V")

            children.append(verb)

        # ----------------------------------------------------
        # OPTIONAL OBJECT NP
        # ----------------------------------------------------

        np_start = self.save()

        np = self.parse_NP()

        if np is not None:

            children.append(np)

        else:

            self.restore(np_start)

        # ----------------------------------------------------
        # OPTIONAL PREPOSITIONAL PHRASES
        # ----------------------------------------------------

        while self.match("PREP"):

            pp_start = self.save()

            pp = self.parse_PP()

            if pp is not None:

                children.append(pp)

            else:

                self.restore(pp_start)

                break

        return Node(
            "VP",
            children
        )

    # ========================================================
    # PP
    #
    # PP -> PREP NP
    # ========================================================

    def parse_PP(self):

        start = self.save()

        self.debug(
            "    [PP] PREP -> NP"
        )

        prep = self.consume("PREP")

        if prep is None:

            self.restore(start)

            return None

        np = self.parse_NP()

        if np is None:

            self.restore(start)

            return None

        return Node(
            "PP",
            [
                prep,
                np
            ]
        )

    # ========================================================
    # COMPLETE PARSE
    # ========================================================

    def parse(self):

        self.position = 0

        tree = self.parse_S()

        # ----------------------------------------------------
        # Ensure EVERYTHING was consumed
        # ----------------------------------------------------

        if tree is not None:

            # Ignore final punctuation.
            #
            # Example:
            # "The boy runs."
            #
            # The grammar parses:
            # The boy runs
            #
            # and punctuation is attached separately.

            punctuation_nodes = []

            while self.match("PUNCT"):

                punctuation = self.consume(
                    "PUNCT"
                )

                punctuation_nodes.append(
                    punctuation
                )

            if (
                self.position
                ==
                len(self.tokens)
            ):

                if punctuation_nodes:

                    tree.children.extend(
                        punctuation_nodes
                    )

                return tree

        return None

    # ========================================================
    # ERROR REPORT
    # ========================================================

    def print_error(self):

        print("\n")
        print("=" * 75)
        print("PARSING ERROR")
        print("=" * 75)

        position = self.farthest_position

        if position < len(self.tokens):

            print(
                "\nParser stopped near:"
            )

            print(
                f"Token:    {self.tokens[position]}"
            )

            print(
                f"Category: {self.categories[position]}"
            )

            print(
                f"Position: {position + 1}"
            )

        else:

            print(
                "\nParser reached the end of input."
            )

        if self.farthest_expected:

            print(
                "\nExpected:"
            )

            for item in sorted(
                self.farthest_expected
            ):

                print(
                    f"  - {item}"
                )

        if position < len(self.tokens):

            print(
                "\nRemaining input:"
            )

            print(
                " ".join(
                    self.tokens[position:]
                )
            )

    # ========================================================
    # TREE PRINTER
    # ========================================================

    def print_tree(
        self,
        node,
        prefix="",
        is_last=True,
        root=True
    ):

        # ----------------------------------------------------
        # ROOT
        # ----------------------------------------------------

        if root:

            print(node.label)

        else:

            connector = (
                "└── "
                if is_last
                else
                "├── "
            )

            print(
                prefix +
                connector +
                node.label
            )

        # ----------------------------------------------------
        # CHILD PREFIX
        # ----------------------------------------------------

        if root:

            child_prefix = ""

        else:

            if is_last:

                child_prefix = (
                    prefix +
                    "    "
                )

            else:

                child_prefix = (
                    prefix +
                    "│   "
                )

        # ----------------------------------------------------
        # CHILDREN
        # ----------------------------------------------------

        for index, child in enumerate(
            node.children
        ):

            last = (
                index ==
                len(node.children) - 1
            )

            self.print_tree(
                child,
                child_prefix,
                last,
                root=False
            )


# ============================================================
# 7. PRINT TOKEN TABLE
# ============================================================

def print_token_table(
    tagged_tokens,
    categories
):

    print("\n")
    print("=" * 75)
    print("TOKEN ANALYSIS")
    print("=" * 75)

    print(
        f"{'No.':<5}"
        f"{'Token':<22}"
        f"{'POS':<12}"
        f"{'Grammar Category':<20}"
    )

    print("-" * 75)

    for index, (
        (word, pos),
        category
    ) in enumerate(
        zip(
            tagged_tokens,
            categories
        ),
        start=1
    ):

        print(
            f"{index:<5}"
            f"{word:<22}"
            f"{pos:<12}"
            f"{category:<20}"
        )


# ============================================================
# 8. PRINT GRAMMAR
# ============================================================

def print_grammar():

    print("\n")
    print("=" * 75)
    print("GRAMMAR")
    print("=" * 75)

    rules = [

        "S         -> CLAUSE",

        "CLAUSE    -> NP VP",
        "CLAUSE    -> NP VP CONJ CLAUSE",

        "NP        -> PRON",
        "NP        -> PROPN",
        "NP        -> DET NOMINAL",
        "NP        -> NOMINAL",
        "NP        -> NUM NOMINAL",

        "NOMINAL   -> ADJ_PHRASE NBAR",
        "NOMINAL   -> NBAR",

        "NBAR      -> N",
        "NBAR      -> N PP*",

        "ADJ_PHRASE -> ADJ",
        "ADJ_PHRASE -> ADV ADJ",

        "VP        -> V",
        "VP        -> V NP",
        "VP        -> V PP",
        "VP        -> V NP PP",

        "VP        -> AUX V",
        "VP        -> AUX V NP",
        "VP        -> AUX V PP",
        "VP        -> AUX V NP PP",

        "VP        -> AUX ADJ_PHRASE",

        "PP        -> PREP NP"
    ]

    for rule in rules:

        print(rule)


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 75)
    print("             ROBUST TOP-DOWN ENGLISH PARSER")
    print("=" * 75)

    print(
        "\nEnter an English sentence."
    )

    print(
        "The program will tokenize it, "
        "POS-tag it, classify the words, "
        "perform top-down parsing, and "
        "build the complete parse tree."
    )

    print_grammar()

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)

    sentence = input(
        "Enter sentence: "
    ).strip()

    if not sentence:

        print(
            "\nNo sentence entered."
        )

        return

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    try:

        tokens = word_tokenize(
            sentence
        )

    except Exception as error:

        print(
            "\nTokenization failed:"
        )

        print(error)

        return

    # --------------------------------------------------------
    # POS TAGGING
    # --------------------------------------------------------

    try:

        tagged_tokens = pos_tag(
            tokens
        )

    except Exception as error:

        print(
            "\nPOS tagging failed:"
        )

        print(error)

        return

    # --------------------------------------------------------
    # CREATE PARSER
    # --------------------------------------------------------

    parser = TopDownParser(
        tokens,
        tagged_tokens,
        verbose=True
    )

    # --------------------------------------------------------
    # TOKEN INFORMATION
    # --------------------------------------------------------

    print_token_table(
        tagged_tokens,
        parser.categories
    )

    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("TOP-DOWN PARSING / DERIVATION")
    print("=" * 75)

    tree = parser.parse()

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if tree is not None:

        print("\n")
        print("=" * 75)
        print("PARSING SUCCESSFUL")
        print("=" * 75)

        print(
            "\nComplete Parse Tree:\n"
        )

        parser.print_tree(tree)

        print("\n")
        print("=" * 75)
        print("TREE COMPLETE")
        print("=" * 75)

        print(
            "\nThe tree was constructed from the "
            "start symbol S and expanded recursively "
            "until the terminal tokens were reached."
        )

    # --------------------------------------------------------
    # FAILURE
    # --------------------------------------------------------

    else:

        print("\n")
        print("=" * 75)
        print("PARSING FAILED")
        print("=" * 75)

        parser.print_error()


# ============================================================
# 10. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()