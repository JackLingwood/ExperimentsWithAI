
def print_red_text(text):
    print("\033[31m" + text + "\033[0m")

def print_blue_text(text):
    print("\033[34m" + text + "\033[0m")

def print_yellow_text(text):
    print("\033[33m" + text + "\033[0m")

def print_green_text(text):
    print("\033[32m" + text + "\033[0m")

def note(text):
    print_yellow_text("\n" + text + "\n")


def heading(h):     
      print_red_text("\n" + "="*len(h))
      print_red_text(h.upper())  
      print_red_text("="*len(h) + "\n")     


def heading2(heading,obj=""):
    print()
    print_green_text(heading.upper())    
    print(obj)

def heading3(heading,obj):
    print("\033[34m" + heading + "\033[0m", end=' ')
    print(obj)

def clearConsole():
       import os
       os.system('cls' if os.name == 'nt' else 'clear')


def setCurrentDirectory(file):
    import os
    current_directory = os.path.dirname(os.path.abspath(file))
    os.chdir(current_directory)
    print(f"Current working directory set to: {current_directory}")

    

def print_ascii_table(data):
    from colorama import Fore, Style, init
    init(autoreset=True)

    if not data or not all(isinstance(row, list) for row in data):
        print("Invalid data format")
        return

    # Determine column widths
    col_widths = [
        max(len(str(row[i])) for row in data) + 2  # +2 for padding
        for i in range(len(data[0]))
    ]

    # Detect numeric columns for right alignment
    is_numeric = [all(isinstance(row[i], (int, float)) for row in data[1:]) for i in range(len(data[0]))]

    # Border drawing characters
    H, V = "─", "│"
    TL, TS, TR = "┌", "┬", "┐"
    ML, MS, MR = "├", "┼", "┤"
    BL, BS, BR = "└", "┴", "┘"

    def build_border(left, sep, right):
        return left + sep.join(H * w for w in col_widths) + right

    def format_cell(content, width, align_right):
        text = str(content)
        if align_right:
            return f" {text:>{width-2}} "
        else:
            return f" {text:<{width-2}} "

    # Top border
    print(build_border(TL, TS, TR))

    # Header row with color and bold
    header = data[0]
    header_line = V + V.join(
        f"{Style.BRIGHT + Fore.CYAN}{format_cell(header[i], col_widths[i], False)}{Style.RESET_ALL}"
        for i in range(len(header))
    ) + V
    print(header_line)

    # Middle border
    print(build_border(ML, MS, MR))

    # Data rows with alignment and no color
    for row in data[1:]:
        row_line = V + V.join(
            format_cell(row[i], col_widths[i], is_numeric[i])
            for i in range(len(row))
        ) + V
        print(row_line)

    # Bottom border
    print(build_border(BL, BS, BR))


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



