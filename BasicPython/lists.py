
Participants = ['John', 'Leila', 'Gregory', 'Cate']
# This code demonstrates the use of lists in Python.
# Lists can hold different data types and can be modified.
print("Participants = ['John', 'Leila', 'Gregory', 'Cate'] prints as ")
print(Participants)
print("Participants[1] as " + Participants[1])
print("Participants[-2] as " + Participants[-2])  # Accessing the second last element
print("Participants[3] = 'Maria', now we have it as ")
Participants[3] = 'Maria'
print(Participants) 
print("Doing del Participants[2]")
del Participants[2]
print(Participants)

print("Doing Participants.append(\"Craig\")")
Participants.append("Craig")
print(Participants)

print("Participants[-1] = 'Dwayne'")    
Participants[-1] = 'Dwayne'
print(Participants)

print("Participants.extend(['George','Catherine'])")
Participants.extend(['George','Catherine'])
print(Participants)
print('The first participant is ' + Participants[0] + '.')

# This code demonstrates the use of lists in Python.
# Lists can hold different data types and can be modified.
list_1 = [1,"John", 54.423]
print('[1,"John", 54.423] as ', list_1)

print("len('Dolphin') prints as")
print(len('Dolphin'))
print("len(Participants) prints as")
print(Participants)  # Display the current list of participants
print("len(Participants) prints as",len(Participants))