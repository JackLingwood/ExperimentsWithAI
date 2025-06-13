# This code demonstrates the use of lists in Python.
# Lists can hold different data types and can be modified.#     return result

Participants = ['John', 'Leila', 'Gregory', 'Cate']
print("Participants = ['John', 'Leila', 'Gregory', 'Cate'] prints as ")
print(Participants)


list_1 = [1,"John", 54.423]
print('[1,"John", 54.423] as '  )
print(list_1)

print("Participants[1] as " + Participants[1])
print("Participants[-2] as " + Participants[2])
Participants[3] = 'Maria'
print("Participants[3] = 'Maria', was have it as ")
print(Participants)

print("Doing del Participants[2]")
del Participants[2]
print(Participants)

print("Doing Participants.append(\"Craig\")") 
Participants.append("Craig")
Participants

Participants[-1] = 'Dwayne'
Participants

Participants.extend(['George','Catherine'])
Participants

print('The first participant is ' + Participants[0] + '.')

len('Dolphin')

len(Participants)