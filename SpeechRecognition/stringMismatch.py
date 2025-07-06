import re
import difflib
from colorama import Fore, Style, init

init(autoreset=True)

def tokenize(text):
    # Tokenize words and punctuation separately
    return re.findall(r"\w+|[^\w\s]", text)

def highlight_differences_diff_based(str1, str2):
    tokens1 = tokenize(str1)
    tokens2 = tokenize(str2)

    matcher = difflib.SequenceMatcher(None, [t.lower() for t in tokens1], [t.lower() for t in tokens2])

    result = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.extend(tokens2[j1:j2])  # No highlight
        elif tag in ("replace", "delete", "insert"):
            for token in tokens2[j1:j2]:
                result.append(f"{Fore.RED}{token}{Style.RESET_ALL}")

    # Reconstruct into readable text
    print("\nOriginal Text:\n")
    print(str1)

    print("\nCompared Text (mismatches in red):\n")
    print(" ".join(result))


# Example usage
str1 = """My name is Ivan and I am excited to have you as part of our learning community!
Before we get started, I’d like to tell you a little bit about myself. I’m a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production,
with a focus on audio, IT, and communications"""

str2 = """my name is Yvonne and I am excited to have you as part of our Learning Community before we get started I'd like to tell you a little bit about myself I'm a sound engineer turn data scientist curious about machine learning and artificial intelligence my professional background is primarily in media production with a focus on audio it and Communications"""

highlight_differences_diff_based(str1, str2)