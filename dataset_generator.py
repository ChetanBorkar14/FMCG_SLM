import pandas as pd
import random
import csv

def generate_transactions(num_transactions=10000, output_file='transactions.csv'):
    categories = {
        'dairy': ['milk', 'butter', 'cheese', 'yogurt'],
        'snacks': ['chips', 'biscuits', 'popcorn', 'chocolate'],
        'beverages': ['tea', 'coffee', 'soft_drinks', 'juice'],
        'household': ['detergent', 'soap', 'shampoo', 'conditioner', 'toothpaste', 'toilet_paper']
    }
    
    # Define realistic co-occurrence rules
    rules = [
        (['milk'], ['bread', 'butter']),
        (['tea'], ['sugar', 'biscuits']),
        (['shampoo'], ['conditioner', 'soap']),
        (['chips'], ['soft_drinks']),
        (['coffee'], ['sugar', 'milk'])
    ]

    transactions = []
    
    for _ in range(num_transactions):
        basket = set()
        
        # Decide how many base categories to pick from
        num_categories = random.randint(1, 3)
        picked_cats = random.sample(list(categories.keys()), num_categories)
        
        for cat in picked_cats:
            items = random.sample(categories[cat], random.randint(1, 2))
            basket.update(items)
            
        # Apply rules to simulate realistic co-occurrence
        for antecedents, consequents in rules:
            if all(ant in basket for ant in antecedents):
                if random.random() < 0.75: # 75% chance to apply rule
                    basket.update(consequents)
                    
        # Filter out random "sugar" and "bread" since they might not be in categories list
        if 'sugar' in basket and 'sugar' not in sum(categories.values(), []):
            basket.add('sugar')
        if 'bread' in basket and 'bread' not in sum(categories.values(), []):
            basket.add('bread')

        # Limit to 10 items max
        basket = list(basket)[:10]
        if basket:
            transactions.append(basket)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(transactions)
        
    print(f"Generated {len(transactions)} synthetic transactions into {output_file}.")

if __name__ == "__main__":
    generate_transactions()
