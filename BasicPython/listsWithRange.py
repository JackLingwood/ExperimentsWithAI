print("Lists with range")
# This code demonstrates the use of the range function to create lists in Python.
# The range function generates a sequence of numbers, which can be converted to a list.
# The range function can take one, two, or three arguments: start, stop, and step.
# The start argument is inclusive, while the stop argument is exclusive.
# The step argument determines the increment between each number in the sequence.
print("l1 = range(0,10) prints as")
l1 = range(0,10)
print(l1)

print("list(range(0,10)) prints as")
print(list(range(0,10)))  # Generates a list of numbers from 0 to 9


l2 = list(range(1,10))
print("l2 = list(range(1,10)) prints as", l2)


l3 = range(3,7)
print("l3 = range(3,7) prints as", l3)


print("list(range(3,7)) prints as",list(range(3,7)))
print("list(range(1,10,2)) prints as",list(range(1,10,2)))  # Generates a list of odd numbers from 1 to 9
print("list(range(0,10,2)) prints as", list(range(0,10,2)))  # Generates a list of even numbers from 0 to 8
print("list(range(10,0,-1)) prints as", list(range(10,0,-1)))  # Generates a list of numbers from 10 to 1 in reverse order
# This code demonstrates the use of the range function to generate a sequence of numbers.
print("Printing powers of 2 using range:")
# The range function can be used to generate a sequence of numbers, which can be used in loops.
# This code demonstrates the use of the range function to generate a sequence of numbers.
# The range function can take one, two, or three arguments: start, stop, and step.
print("Printing powers of 2 using range:")
print("for n in range(10):")
print("    print(2**n, end=' ')")
for n in range(10):
    print(2**n, end=" ")

# This code demonstrates the use of the range function to generate a sequence of numbers.
print("\nPrinting even numbers from 0 to 20 using range:")
print("for x in range(20):")
print("    if x%2 == 0:")
print("        print(x, end=' ')")
print("    else:")
print("        print('Odd', end=' ')")
print("Printing even numbers from 0 to 20 using range:")
for x in range(20):
    if x%2 == 0:
        print(x,end=" ")
    else:
        print("Odd", end=" ")

# This code demonstrates the use of lists with the range function in Python.
print("\nUsing range with lists:")
# The range function can be used to create lists, which can then be iterated over.
# This code demonstrates the use of lists with the range function in Python.
print("x = [0,1,2] prints as")
x = [0,1,2]
print(x)

print("Iterating over x using for loop:")
print("for item in x:")
print("    print(item, end=' ')")
print("Iterating over x using for loop:")
for item in x:
    print(item, end = " ")

print("\nIterating over x using range:")
print("for item in range(len(x)):")
print("    print(x[item], end=' ')")
print("Iterating over x using range:")

for item in range(len(x)):
    print(x[item],end = " ")