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


# Example usage
data = [
    ["Name", "Age", "City"],
    ["Alice", 30, "New York"],
    ["Bob", 24, "London"],
    ["Charlie", 35, "Paris"]
]

print_ascii_table(data)
