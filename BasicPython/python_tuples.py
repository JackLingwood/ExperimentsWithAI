
print("Demonstrating Python Tuples")
# Tuples are immutable sequences in Python, meaning they cannot be changed after creation.
# They can hold multiple items and are defined using parentheses.
# This code demonstrates the use of tuples in Python.
# A tuple can be created by placing items inside parentheses, separated by commas.
# Tuples can hold different data types and can be accessed using indexing.
# Tuples are often used to group related data together.
# Tuples can be created with or without parentheses.
# Demonstrating tuple creation and access
# Creating a tuple with parentheses
# and accessing its elements using indexing
# Tuples are immutable, meaning their elements cannot be changed after creation.
# However, we can create a new tuple with modified values.
x = (40, 41, 42)
print("x = (40, 41, 42) is a tuple and prints as ",x)
print("x[0] prints as " + str(x[0]))  # Accessing the first element of the tuple
print("x[1] prints as " + str(x[1]))  # Accessing the second element of the tuple
print()
y = 50,51,52
print("y=50,51,52 is a tuple and prints as ", y)
print("y[0] prints as " + str(y[0]))  # Accessing the first element of the tuple
print("y[1] prints as " + str(y[1]))  # Accessing the second element of the tuple
print("y[2] prints as " + str(y[2]))  # Accessing the third element of the tuple
print("y[0] + y[1] prints as " + str(y[0] + y[1]))  # Adding the first two elements of the tuple
print()
# Tuples are immutable, so we cannot change their elements
# However, we can create a new tuple with modified values
print("Creating a new tuple with modified values")
x = (x[0] + 10, x[1] + 10, x[2] + 10)
print("New x tuple after modification: ", x)
# Demonstrating tuple unpacking
# Unpacking the tuple into variables
print()
print("Unpacking the tuple x into variables a, b, c")
print("a,b,c = 1,4,6")
a,b,c = 1,4,6
print("a = " + str(a))
print("b = " + str(b))
print("c = " + str(c))
print("Accessing the first element of the tuple x using x[0] prints as " + str(x[0]))  # Accessing the first element of the tuple
print("Accessing the second element of the tuple x using x[1] prints as " + str(x[1]))  # Accessing the second element of the tuple
print("Accessing the third element of the tuple x using x[2] prints as " + str(x[2]))  # Accessing the third element of the tuple
print()

# Demonstrating tuple packing and unpacking
List_1 = [x,y]
print("List_1 = [x,y]",List_1)


print(  "(age, years_of_school) = \"30,17\".split(',')"    )
(age, years_of_school) = "30,17".split(',')
print("age = " + age)
print("years_of_school = " + years_of_school)
age = int(age)  # Convert age to integer
years_of_school = int(years_of_school)  # Convert years_of_school to integer
print("Converted age and years_of_school to integers:")

# Demonstrating a function that calculates area and perimeter of a square
# and returns them as a tuple
def square_info(x):
    A = x ** 2
    P = 4 * x
    print("Area and Perimeter:")
    return A,P

print("square_info(3) returns the area and perimeter of a square with side length 3:",square_info(3)) 