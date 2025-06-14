dict = { 'k1': "cat", 'k2': "dog", "k3": "mouse", "k4":"fish" }
print("dict = { 'k1': 'cat', 'k2': 'dog', 'k3': 'mouse', 'k4':'fish' } =", dict)
# This code demonstrates the use of dictionaries in Python.
# Dictionaries are mutable, unordered collections of key-value pairs.
# They can hold different data types and can be modified.
# Dictionaries are defined using curly braces {} and consist of key-value pairs.
# Each key is unique and maps to a value.
# Accessing values in a dictionary is done using the keys.
# This code demonstrates the use of dictionaries in Python.
# Dictionaries can be used to store related data together.

print("Accessing values in the dictionary using keys:")
print("dict['k1'] prints as " + dict['k1'])  # Accessing the value associated with key 'k1'
print("dict['k3'] prints as " + dict["k3"])  # Accessing the value associated with key 'k3'
# Adding a new key-value pair to the dictionary
print("dict['k5'] = 'parrot' adds a new key-value pair to the dictionary")
dict['k5'] = "parrot"
print(dict)

dict['k2'] = "squirrel"
print("dict['k2'] = 'squirrel' modifies the value associated with key 'k2'")
# Modifying the value associated with key 'k2'
print(dict)


dep_workers = { 'dep_1' : 'Peter', 'dep_2' : ['Jennifer','Michael','Tommy']}
print("dep_workers = { 'dep_1' : 'Peter', 'dep_2' : ['Jennifer','Michael','Tommy'] }")
print(dep_workers)  

# This code demonstrates the use of dictionaries in Python.

print("Team = {}")
Team = {}
Team['Point Guard'] = 'Dirk'
Team['Shooting Guard'] = 'Al'
Team['Small Forward'] = 'Sean'
Team['Power Forward'] = 'Alexander'
Team['Center'] = 'Hector'
print(Team)

print(Team['Center'])

print(Team.get('Small Forward'))

# This code demonstrates how to access values in a dictionary using both the get method and direct indexing.
print("Accessing values in the Team dictionary:")
print(Team.get('Coach', 'Not Found'))  # Using get method with a default value
# Accessing a key that does not exist will return None or a default value if provided
print(Team.get('Coach'))

# This code will give an error. It is better to user dict.get('key', default_value) to avoid KeyError
# print(Team['Coach'])  # This will raise a KeyError if 'Coach' does not exist
print(Team["Coach"])