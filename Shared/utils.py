
def print_red_text(text):
    print("\033[31m" + text + "\033[0m")

def heading(h):     
      print_red_text("\n" + "="*len(h))
      print_red_text(h.upper())  
      print_red_text("="*len(h) + "\n")     

def clearConsole():
       import os
       os.system('cls' if os.name == 'nt' else 'clear')


def setCurrentDirectory(file):
    import os
    current_directory = os.path.dirname(os.path.abspath(file))
    os.chdir(current_directory)
    print(f"Current working directory set to: {current_directory}")       