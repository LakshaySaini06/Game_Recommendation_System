import csv
import random

# Configuration
NUM_SAMPLES = 5000
OUTPUT_FILE = "game_recommendation_training_data.csv"

# Domain Data
REGIONS = ['NA', 'EU', 'ASIA', 'SA', 'OCE', 'AFR']
GENDERS = ['Male', 'Female', 'Non-binary', 'Other']
GENRES = ['FPS', 'RPG', 'Simulation', 'Strategy', 'Action', 'Adventure', 'Puzzle', 'JRPG', 'Battle Royale', 'Horror']

def generate_preference(age, gender, region, genre):
    """
    Calculates a 'rating' (1-5) based on the same rules encoded in the JS ML Engine.
    This creates a dataset that 'proves' the current model's logic.
    """
    score = 3.0 # Base neutral score
    noise = random.uniform(-0.5, 0.5) # Add randomness

    # Age Bias
    if age < 18:
        if genre in ['FPS', 'Battle Royale', 'Action', 'Social']: score += 1.5
        if genre in ['Strategy', 'Puzzle']: score -= 0.5
    elif age < 35: # Adult
        if genre in ['Story', 'RPG', 'Action-Adventure', 'Adventure']: score += 1.2
    else: # Mature
        if genre in ['Strategy', 'Simulation', 'Puzzle', 'Classic']: score += 1.5
        if genre in ['FPS', 'Battle Royale']: score -= 1.0

    # Gender Bias (Statistical, mirroring the JS engine)
    if gender == 'Male':
        if genre in ['FPS', 'Action', 'Strategy', 'Competitive']: score += 0.8
    elif gender == 'Female':
        if genre in ['Simulation', 'Story', 'Puzzle', 'Creative']: score += 1.0
        if genre in ['FPS']: score -= 0.2

    # Region Bias
    if region == 'ASIA':
        if genre in ['JRPG', 'RPG', 'Strategy']: score += 1.2
    elif region == 'NA':
        if genre in ['FPS', 'Action', 'Sports']: score += 0.8
    elif region == 'EU':
        if genre in ['Simulation', 'Strategy']: score += 0.8

    # Clamp logic
    final_score = int(round(score + noise))
    return max(1, min(5, final_score))

# Generate Data
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['User_ID', 'Age', 'Gender', 'Region', 'Game_Genre', 'Interaction_Rating'])

    for i in range(NUM_SAMPLES):
        age = random.randint(13, 65)
        gender = random.choice(GENDERS)
        region = random.choice(REGIONS)
        genre = random.choice(GENRES)
        
        rating = generate_preference(age, gender, region, genre)
        
        writer.writerow([f"U_{1000+i}", age, gender, region, genre, rating])

print(f"Successfully generated {NUM_SAMPLES} training samples in '{OUTPUT_FILE}'")
