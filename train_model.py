import csv
import json
from collections import defaultdict

INPUT_FILE = "game_recommendation_training_data.csv"

# Buckets matching our JS logic
def get_age_group(age):
    if age < 18: return 'young'
    if age < 35: return 'adult'
    return 'mature'

def train():
    # Accumulate ratings: key -> [ratings]
    # keys will be (category, value, genre)
    # e.g. ('age', 'young', 'FPS')
    data_points = defaultdict(list)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = int(row['Age'])
            gender = row['Gender'].lower() # JS uses lowercase keys for gender
            region = row['Region']
            genre = row['Game_Genre']
            rating = float(row['Interaction_Rating'])
            
            # Age Feature
            age_group = get_age_group(age)
            data_points[('age', age_group, genre)].append(rating)
            
            # Gender Feature
            data_points[('gender', gender, genre)].append(rating)
            
            # Region Feature
            data_points[('region', region, genre)].append(rating)

    # Calculate Weights
    # We'll treat the weight as (AverageRating - NeutralScore)
    # Neutral Score is ~3.0
    NEUTRAL_SCORE = 3.0
    
    learned_weights = {
        'age': defaultdict(dict),
        'gender': defaultdict(dict),
        'region': defaultdict(dict)
    }
    
    for (category, key, genre), ratings in data_points.items():
        avg_rating = sum(ratings) / len(ratings)
        weight = avg_rating - NEUTRAL_SCORE
        
        # Only keep significant weights to keep JSON small
        if abs(weight) > 0.1:
            learned_weights[category][key][genre] = round(weight, 2)
            
    # Output as JSON
    print(json.dumps(learned_weights, indent=2))

if __name__ == "__main__":
    train()
