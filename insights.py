from mba import mba_engine

class InsightsGenerator:
    def __init__(self):
        pass
        
    def generate_insight_report(self):
        rules = mba_engine.get_rules()
        top_selling = mba_engine.get_top_selling()
        
        insights = []
        insights.append(f"Top selling products: {', '.join(top_selling)}")
        
        has_cross_sell = False
        for rule in rules[:10]: # Process top 10 rules
            ant = ", ".join(rule['antecedents'])
            cons = ", ".join(rule['consequents'])
            conf = int(rule['confidence'] * 100)
            lift = rule['lift']
            
            if lift > 1.2:
                insights.append(f"High lift observed between {ant} and {cons} (Lift: {lift:.1f}) → strong cross-sell opportunity!")
                has_cross_sell = True
            
            insights.append(f"Customers who buy {ant} also buy {cons} (Confidence: {conf}%).")
            
        return insights

    def answer_query(self, query: str) -> str:
        query = query.lower()
        rules = mba_engine.get_rules()
        top_selling = mba_engine.get_top_selling()
        
        if "top" in query or "best" in query or "selling" in query:
            return f"The top selling FMCG products are: {', '.join(top_selling)}."
            
        if "together" in query or "bought with" in query or "frequent" in query:
            if not rules:
                return "I don't have enough data on what is bought together yet."
            r = rules[0]
            ant = " and ".join(r['antecedents'])
            cons = " and ".join(r['consequents'])
            return f"{ant.capitalize()} and {cons} are frequently bought together (Confidence: {int(r['confidence']*100)}%, Support: {r['support']})."
            
        if "stock" in query or "inventory" in query:
            if not top_selling:
                return "Cannot determine stocking suggestions without data."
            return f"You should maintain high inventory for your top movers: {', '.join(top_selling)}."
            
        if "cross-sell" in query or "cross sell" in query or "opportunity" in query:
            high_lift_rules = [r for r in rules if r['lift'] > 1.2]
            if high_lift_rules:
                best = high_lift_rules[0]
                ant = ", ".join(best['antecedents'])
                cons = ", ".join(best['consequents'])
                return f"Strong cross-selling opportunity: Customers buying {ant} are very likely to buy {cons} (Lift: {best['lift']:.2f}). Consider bundling them."
            else:
                return "No strong cross-selling opportunities detected above the lift threshold."
                
        return "I can answer questions about: top selling products, cross-selling opportunities, stocking suggestions, and frequently bought together items."

insights_generator = InsightsGenerator()
