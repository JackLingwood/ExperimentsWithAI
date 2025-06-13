# demonstrates how we can iterate over a dictionary in Python.
prices = {
    "bag_dynamite" : 4,
    "grenade" : 5,
    "BLT_sandwich" : 2
}
quantity = {
    "bag_dynamite": 6,
    "grenade" : 10,
    "BLT_sandwich" : 4
}

money_spent = 0
for i in quantity:
    money_spent += (prices[i] * quantity[i])    

print(money_spent)