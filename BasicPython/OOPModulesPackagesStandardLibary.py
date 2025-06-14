x = [15.9, 12.4, 64.0]
x.extend([1,2,3])
print(x)

#import module
#import package
# package = library
# Python standard libary = collection of functions embedded into base version of Python, eg, len

import math
import math as m  # Importing the math module with an alias

y = math.sqrt(9)
print("y = math.sqrt(9)")
print(y)

# This code demonstrates the use of modules and packages in Python.
# Modules are files containing Python code that can be imported and used in other Python programs.
from math import sqrt
sqrt(25)

# This code demonstrates the use of the math module in Python.
# The math module provides mathematical functions and constants.
# It can be imported using the import statement.
# Importing the math module to use its functions
from math import sqrt as s
s(36)

# This code demonstrates the use of the math module in Python.
import math as m
m.sqrt(49)

from math import * # not recommended. May lead to name conflicts
sqrt(64)

# This code demonstrates the use of the math module in Python.
# The math module provides mathematical functions and constants.
# Display help for the sqrt function in the math module
help(math)  

help(math.sqrt)