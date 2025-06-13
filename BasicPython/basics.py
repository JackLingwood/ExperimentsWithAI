# Basic strings in Python
# String can be with single or double quotes in Python
a = "George"  
b = 'George'
y = 10

# You cannot directly combine strings and ints, you to convert the int using str
print( str(y) + " Dollars ")
c = "I'm fine"
d = 'I\'m fine'
e = 'Press "Enter"'
f = 'Red' 'car'

g = 'The red fish '+'is fast'.upper()
# Basic string concatenation and printing
h = 'Red ' + 'car'
print('Red ' + 'car')
print('Red', 'car')
print(3,5)
print(3, 5, 6.9, 7.0, 'car')
print(a,b,c,d,e,f,g,h)


# Basic arithmetic operations in Python
# Addition, subtraction, multiplication, division, modulus, and exponentiation
# python typing is dynamic, so you can assign any type to a variable
i = 1+2
j = 3-5
k = 15/3
l = 16 / 3
m = int(16/3)
n = float(16)/3
o = 16%3
p = 5*3
q = 5*3

print(i, j, k, l, m, n, o, p, q)
# Basic boolean operations in Python


# Python expnentiation operator
import math
y = 5**3

# Equivalents of the above operations
y = math.pow(5, 3)

# Basic comparison operations in Python
y = 5 == 5
print(y)
y = 5 != 5
y = 5 < 10
print(y)
y = 5 > 10
print(y)
y = 5 <= 5
print(y)
y = 5 >= 5
print(y)

# Basic logical operations in Python
# Line continuation in Python is done using the backslash character
# and can be used to split long lines of code for better readability.
a1 = 2.0 * 1.5 + \
5.0 * 2.5
print(a1)

# Simplest function, always need indentation in Python
def five(x):
    x = x + 5
    return x
    
print(five(3))

# Simple Boolean Operators
print("\nSimple Boolean Operators\m\n".upper())

print("True and True is "+(True and True).__str__())
print("True and False is "+(True and False).__str__())
print("False and False is "+(False and False).__str__())
print("True or True is "+(True or True).__str__())
print("True or False is "+(True or False).__str__())
print("False or True is "+(False or True).__str__())
print("not True is "+(not True).__str__())
print("not False is "+(not False).__str__())
print("3 > 5 and 10 <= 20 is "+(3 > 5 and 10 <= 20).__str__())
print("True and not True is "+(True and not True).__str__())
print("False or not True and True is "+(False or not True and True).__str__())
print("True and not True or True is "+(True and not True or True).__str__())
print("5 is 6 is "+(5 is 6).__str__())
print("5 is 5 is "+(5 is 5).__str__())
print("5 is not 6 is "+(5 is not 6).__str__())
print("5 is not 5 is "+(5 is not 5).__str__())
print("5 is 5.0 is "+(5 is 5.0).__str__())
print("5 is not 5.0 is "+(5 is not 5.0).__str__())
print("5 is 5.0 is "+(5 == 5.0).__str__())
print("5 is not 5.0 is "+(5 != 5.0).__str__())
print("5 is 5.0 is "+(5 == 5.0).__str__())
print("5 is not 5.0 is "+(5 != 5.0).__str__())
print("5 is 5.0 is "+(5 == 5.0).__str__())
print("5 is not 5.0 is "+(5 != 5.0).__str__())
print("5 is 5.0 is "+(5 == 5.0).__str__())


# Combining functions

print("Basic function in pythons".upper())

def simple():
    print("My first function")
    
simple()

def plus10(a):
    return a+10

print(plus10(5))

plus10(2)

def plus_ten(a):
    result = a + 10    
    return result

def plus_ten(a):
    result = a + 10    
    print("Outcome")
    return result

plus_ten(2)

def wage(w_hours):
    return w_hours*25

def with_bonus(w_hours):
    return wage(w_hours) + 50

wage(8), with_bonus(8)

def add_10(m):
    if m>= 100:
        m = m + 10
        return m
    else:
        return "Save more!"

add_10(110)

add_10(50)

# Basic Condition Statements
print("Basic Condition Statements".upper()) 
if 5==15/3:
    print ("Hooray!")

if (5==18/3):
    print ("Hooray!")
else:
    print ("Boo!")

if (5 != 3*6):
    print("Hooray!")


# Basic If-Else Statements
print("Basic If-Else Statements".upper())
def compare_to_five(x):
    if x > 5:
        return "Greater"
    elif x < 5:
        return "Less"
    else:
        return "Equal"
def compare_to_five(y):
    if y>5:
        return "Greater"
    elif y<5:
        return "Less"
    else:
        return "Equal"

print(compare_to_five(10))
print(compare_to_five(-3))
print(compare_to_five(3))
print(compare_to_five(5))

# Some basic functions and such
x = 2
if x>4:
    print("Correct")
else:
    print("Incorrect")

def subtract_bc(a,b,c):
    result = a - b*c
    print('Parameter a equals', a)
    print('Parameter b equals', b)
    print('Parameter c equals', c)
    return result

subtract_bc(10,5,4)
subtract_bc(10,3,2)
subtract_bc(b=3,a=10,c=2)

print('\nSome basic operatores and functions\n'.upper())
print("type('Hello') = " + str(type('Hello')))
print("type(10) = " + str(type(10)))
print("type(10.0) = "+ str(10.0))
print("type(True) = "+ str(type(True)))
print("type(False) = "+ str(type(False)))
print("str(500) = "+ str(500).__str__())
print("max(10,20,30,40) = "+ str(max(10,20,30,40)).__str__())
print("min(10,20,30,40) = "+ str(min(10,20,30,40)).__str__())
print("abs(-20) = "+ str(abs(-20)).__str__())
print("list_1 = [1,2,3,4]")
list_1 = [1,2,3,4]
print("sum(list_1) = "+ str(sum(list_1)).__str__())
print("round(3.555,2) = "+ str(round(3.555,2)).__str__())
print("round(3.2) = "+ str(round(3.2)).__str__())
print("2**10 = "+ str(2**10).__str__())
print("pow(2,10) = "+ str(pow(2,10)).__str__())
print("len('Mathematics') = "+ str(len('Mathematics')).__str__())
