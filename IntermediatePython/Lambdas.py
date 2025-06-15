
print("\nFRUITY LAMBDAS EXAMPLE\n")

fruit = ["apple", "banana", "cherry", "date", "elderberry"]

badfruit = ["banana", "date"]

# data['review_no_stopwords'] = data['review_lowercase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))



data = []

# Using a lambda function to filter out bad fruits

print("Fruit: ",fruit)
data = {
    'fruit': fruit,
    'good_fruit': list(map(lambda x: x if x not in badfruit else None, fruit))
}
print("Good Fruit: ", data['good_fruit'])
print()
print("data: ")
print(data)


# goodfruit = fruit.apply(lambda x: x if x not in badfruit else None)
# print("Good Fruit using apply: ", goodfruit)
print("\nUsing a lambda function to filter out bad fruits with list comprehension:")
goodfruit_list = [x for x in fruit if x not in badfruit]
print("Good Fruit using list comprehension: ", goodfruit_list)
print("\nUsing a lambda function to filter out bad fruits with filter:")
goodfruit_filter = list(filter(lambda x: x not in badfruit, fruit))
print("Good Fruit using filter: ", goodfruit_filter)
print("\nUsing a lambda function to filter out bad fruits with map:")
goodfruit_map = list(map(lambda x: x if x not in badfruit else None, fruit))
print("Good Fruit using map: ", goodfruit_map)
print("\nUsing a lambda function to filter out bad fruits with a for loop:")
goodfruit_for = []
for x in fruit:
    if x not in badfruit:
        goodfruit_for.append(x)
print("Good Fruit using for loop: ", goodfruit_for)
print("\nUsing a lambda function to filter out bad fruits with a while loop:")
goodfruit_while = []
i = 0
while i < len(fruit):
    if fruit[i] not in badfruit:
        goodfruit_while.append(fruit[i])
    i += 1
print("Good Fruit using while loop: ", goodfruit_while)
print("\nUsing a lambda function to filter out bad fruits with a generator expression:")
goodfruit_gen = (x for x in fruit if x not in badfruit)
print("Good Fruit using generator expression: ", list(goodfruit_gen))
print("\nUsing a lambda function to filter out bad fruits with a set comprehension:")
goodfruit_set = {x for x in fruit if x not in badfruit}
print("Good Fruit using set comprehension: ", goodfruit_set)
print("\nUsing a lambda function to filter out bad fruits with a dictionary comprehension:")
goodfruit_dict = {x: x for x in fruit if x not in badfruit}
print("Good Fruit using dictionary comprehension: ", goodfruit_dict)
print("\nUsing a lambda function to filter out bad fruits with a numpy array:")
import numpy as np
goodfruit_np = np.array([x for x in fruit if x not in badfruit])
print("Good Fruit using numpy array: ", goodfruit_np)
