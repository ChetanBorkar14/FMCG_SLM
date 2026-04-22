import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import csv

class MarketBasketAnalyzer:
    def __init__(self, dataset_path='transactions.csv'):
        self.dataset_path = dataset_path
        self.frequent_itemsets = None
        self.rules = None
        self.top_selling = []
        
    def load_and_mine(self, min_support=0.01, min_confidence=0.2):
        print("Loading transactions and running FP-Growth...")
        # Load custom synthetic transactions
        transactions = []
        with open(self.dataset_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    transactions.append(row)
                    
        # Calculate top selling manually
        item_counts = {}
        for basket in transactions:
            for item in basket:
                item_counts[item] = item_counts.get(item, 0) + 1
                
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        self.top_selling = [item[0] for item in sorted_items[:5]]

        # Encode and run FP-Growth
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        self.frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
        if len(self.frequent_itemsets) > 0:
            self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            # Additional cleanup rules DataFrame
            self.rules['antecedents'] = self.rules['antecedents'].apply(lambda x: list(x))
            self.rules['consequents'] = self.rules['consequents'].apply(lambda x: list(x))
            self.rules = self.rules.sort_values(['lift', 'confidence'], ascending=[False, False])
        else:
            self.rules = pd.DataFrame()
            
        print(f"Mined {len(self.frequent_itemsets)} frequent itemsets and {len(self.rules)} rules.")
        
    def get_rules(self):
        if self.rules is None or self.rules.empty:
            return []
            
        rules_list = []
        for _, row in self.rules.head(50).iterrows(): # Top 50 rules
            rules_list.append({
                'antecedents': row['antecedents'],
                'consequents': row['consequents'],
                'support': round(row['support'], 3),
                'confidence': round(row['confidence'], 3),
                'lift': round(row['lift'], 3)
            })
        return rules_list
        
    def get_top_selling(self):
        return self.top_selling

# Create singleton instance
mba_engine = MarketBasketAnalyzer()
