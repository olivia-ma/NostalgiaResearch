##### IMPORTS
import pandas as pd
import requests
import time
import json
from tqdm import tqdm
import re

##### FILE_PATHS
# input_path = '/kaggle/input/input-data/'
input_path ='files/'
raw_game_id_path = f'{input_path}games.csv' # this has app id list to use
master_game_raw_path = f'{input_path}games_master.csv' # getting game data from raw app id list
master_appid_path = f'{input_path}app_ids.txt' # all app ids in master list


###### Data constraints
COVID_YEAR = 2020


########### INPUT DATA HELPER FUNCTIONS ########################################
def get_game_details(appid):
    url = "https://store.steampowered.com/api/appdetails"
    params = {'appids': appid}

    data = safe_request(url, params=params)
    if not data:
        return None

    if str(appid) not in data or not data[str(appid)].get('success'):
        return None

    game_data = data[str(appid)]['data']

    return {
        'appid': appid,
        'name': game_data.get('name'),
        'type': game_data.get('type'),
        'release_date': game_data.get('release_date', {}).get('date'),
        'required_age': game_data.get('required_age'),
        'developers': ', '.join(game_data.get('developers', [])),
        'publishers': ', '.join(game_data.get('publishers', [])),
        'genres': ', '.join([g['description'] for g in game_data.get('genres', [])]),
        'price': game_data.get('price_overview', {}).get('final', 0) / 100 if game_data.get('price_overview') else 0,
        'metacritic_score': game_data.get('metacritic', {}).get('score'),
        'recommendations': game_data.get('recommendations', {}).get('total'),
        'achievements': len(game_data.get('achievements', {}).get('achievements', [])),
    }


def get_steamspy_data(appid):
    url = "https://steamspy.com/api.php"
    params = {'request': 'appdetails', 'appid': appid}

    data = safe_request(url, params=params)
    if not data or 'name' not in data:
        return {}

    # ---------- SAFE TAG PARSING ----------
    tags_raw = data.get("tags", {})

    if isinstance(tags_raw, dict):
        tags = sorted(tags_raw.items(), key=lambda x: x[1], reverse=True)
        tags = [k for k, v in tags[:5]]
    elif isinstance(tags_raw, list):
        tags = tags_raw[:5]
    else:
        tags = []

    tags_str = ", ".join(tags)
    # --------------------------------------

    return {
        'owners_estimate': data.get('owners'),
        'players_forever': data.get('players_forever'),
        'average_playtime': data.get('average_forever'),
        'median_playtime': data.get('median_forever'),
        'tags': tags_str,
        'positive_reviews': data.get('positive'),
        'negative_reviews': data.get('negative'),
    }



import requests, time, random

def safe_request(url, params=None, timeout=10, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)

            # Check HTTP status
            if r.status_code != 200:
                raise Exception(f"HTTP {r.status_code}")

            # Parse JSON (Steam sometimes returns HTML on rate-limit)
            try:
                return r.json()
            except:
                raise Exception("Non-JSON response (rate limit?)")

        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠️ Error: {e}, retrying in {wait}s...")
            time.sleep(wait)

    print("  ❌ Failed after max retries.")
    return None


###################################### INPUT DATA ##########################################
# ==========================================
# STEP 0: GETTING STEAM GAME "App IDs"
# ==========================================
print("Step 0: GETTING STEAM GAME 'App IDs'...")
df = pd.read_csv(f"{raw_game_id_path}", engine='python', on_bad_lines='skip')
test_appids = df.index.to_list()
print(f"RAW - Fetched {len(test_appids)} games list.")
# test_appids = test_appids
# ==========================================
# STEP 1: Getting MASTER GAME LIST  [DATASET 1]
# ==========================================
print("\nStep 1: Getting MASTER GAME LIST  [DATASET 1]...")
all_games = []
MAX_GAMES = 1000
# test_appids = test_appids[2062:]
temp_test = test_appids.copy()
# test_appids = test_appids[7931:]
# df_appids = pd.DataFrame(test_appids)
for appid in tqdm(test_appids):
  if len(all_games) >= MAX_GAMES :
    print(f"REACHED {MAX_GAMES} at appid = {appid}.")
    break
  else:
    # print(f"\nTesting AppID {appid}...")

    # Test Steam Store API
    game_data = get_game_details(appid)
    if game_data:
        # print(f"  ✓ Steam Store: {game_data['name']}")
        pass
    else:
        # print(f"  ✗ Steam Store failed")
        continue

    # ---- checking release year -----
    release_str = game_data.get("release_date") or ""
    m = re.search(r"(\d{4})", release_str)
    if not m:
        # no usable year found → skip
        print(m,len(all_games))
        continue

    year = int(m.group(1))

    # keep only pre-2020 games
    if year >= COVID_YEAR:
        continue

    # store year in the record too (useful later)
    game_data["year"] = year

    # Test SteamSpy API
    time.sleep(1)
    spy_data = get_steamspy_data(appid)
    if spy_data:
        # print(f"  ✓ SteamSpy: {spy_data.get('owners_estimate')} owners")
        game_data.update(spy_data)
        all_games.append(game_data)
    else:
        pass
        # print(f"  ✗ SteamSpy failed")
df_games = pd.DataFrame(all_games)
print(f"Fetched {len(df_games)} pre-COVID games.")
df_pre_covid = df_games
df_pre_covid.to_csv(f'{master_game_raw_path}', index=False)
df_pre_covid['appid'].to_csv(f'{master_appid_path}', index=False, header=False)

print("\n" + "="*50)
print(f"✓ games_master.csv: {len(df_pre_covid)} games")
print(f"✓ app_ids.txt: List of {len(df_pre_covid)} app IDs for Person B")
print("\nColumns available:")
print(df_pre_covid.columns.tolist())
print("\nSample data:")
print(df_pre_covid.head())

def get_steam_reviews(appid, num_reviews=100):
    """
    Fetch reviews for a single Steam game
    Returns list of review dictionaries
    """
    all_reviews = []
    cursor = '*'

    while len(all_reviews) < num_reviews:
        url = "https://store.steampowered.com/appreviews/" + str(appid)
        params = {
            'json': '1',
            'filter': 'recent',
            'language': 'english',
            'cursor': cursor,
            'num_per_page': '100'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            # Check if request was successful
            if data.get('success') != 1:
                print(f"  Failed to get reviews for {appid}")
                break

            # Extract reviews
            reviews = data.get('reviews', [])
            if not reviews:
                break

            for review in reviews:
                all_reviews.append({
                    'appid': appid,
                    'review_text': review['review'],
                    'recommended': review['voted_up'],
                    'playtime_forever': review['author']['playtime_forever'] / 60,  # Convert to hours
                    'playtime_at_review': review['author']['playtime_at_review'] / 60,
                    'timestamp': review['timestamp_created'],
                    'helpful_votes': review['votes_up']
                })

            # Get next page cursor
            cursor = data.get('cursor', None)
            if not cursor:
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

    return all_reviews[:num_reviews]


# === MAIN COLLECTION SCRIPT ===

# Use pre-COVID appids directly from df_pre_covid
pre_covid_appids = df_pre_covid["appid"].dropna().unique().tolist()

print(f"Total pre-COVID games: {len(pre_covid_appids)}")

# (Optional) limit if you want to test first
# pre_covid_appids = pre_covid_appids[:200]

all_reviews = []
failed_games = []

for i, appid in tqdm(enumerate(pre_covid_appids)):
    # print(f"\n[{i+1}/{len(pre_covid_appids)}] Fetching AppID {appid}...")

    reviews = get_steam_reviews(appid, num_reviews=100)

    if reviews:
        all_reviews.extend(reviews)
        # print(f"  ✓ Got {len(reviews)} reviews")
    else:
        failed_games.append(appid)
        print(f"  ✗ Failed")

    # Rate limiting
    time.sleep(1.5)

    # Save checkpoint every 100 games (tune this)
    if (i + 1) % 500 == 0:
        checkpoint_df = pd.DataFrame(all_reviews)
        checkpoint_df.to_csv(f'reviews_checkpoint_{i+1}.csv', index=False)
        print(f"\n  💾 Saved checkpoint: {len(all_reviews)} total reviews")

df_reviews = pd.DataFrame(all_reviews)
df_reviews.to_csv('steam_reviews_raw.csv', index=False)

print("\n" + "="*50)
print("COLLECTION COMPLETE")
print("="*50)
print(f"Total reviews collected: {len(df_reviews)}")
print(f"Unique games: {df_reviews['appid'].nunique()}")
print(f"Failed games: {len(failed_games)}")
print(f"Average reviews per game: {len(df_reviews) / df_reviews['appid'].nunique():.1f}")
print(f"\n✓ Saved to: steam_reviews_raw.csv")

# Show sample data
print("\nFirst few rows:")
print(df_reviews.head())

# Show statistics
print("\nDataset Statistics:")
print(df_reviews.describe())