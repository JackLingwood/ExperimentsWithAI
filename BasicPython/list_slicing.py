Participants = ['John','Leila','Maria','Dwayne','George','Catherine']
print("Participants = ['John','Leila','Maria','Dwayne','George','Catherine'] is ")
print(Participants)

# This code demonstrates the use of lists in Python.
# Lists can hold different data types and can be modified.
print("Participants[1] as " + Participants[1])
print("Participants[-2] as " + Participants[-2])
Participants[3] = 'Maria'
print("Participants[3] = 'Maria', now we have it as ")
print(Participants)

print("Participants[1:3] prints as ")
print(Participants[1:3])  # Slicing the list to get elements from index 1 to 2


print("Participants[:2] prints as ")
print(Participants[:2])  # Slicing the list to get the first two elements

print("Participants[:-2] prints as ")
print(Participants[:-2])  # Slicing the list to get all but the last two elements
print("Participants[4:] prints as ")
print(Participants[4:])


print("Participants.index(\"Maria\") prints as")
print(Participants.index("Maria"))  # Finding the index of 'Maria'

print("Newcomers = ['Joshua', 'Brittany'] prints as")
Newcomers = ['Joshua', 'Brittany']
print(Newcomers)


print("Combing lists")
print("Bigger_List = [Participants, Newcomers] prints as")
Bigger_List = [Participants, Newcomers]
print(Bigger_List)

print("Participants.sort() prints as")
print("Sorting the list Participants in ascending order")
Participants
print(Participants)

print("reversing the list Participants")
Participants.sort()
print("Participants.sort(reverse=True) prints as")
print(Participants)


Numbers = [1,2,3,4,5]
Numbers.sort()
print(Numbers)

print("Sorting the list Numbers in descending order")
Numbers = [1, 2, 3, 4, 5]   
Numbers.sort(reverse=True)  
print(Numbers)