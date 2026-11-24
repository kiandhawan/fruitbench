"""Print the Fish API credit balance (FISH_API_KEY=$(cat ~/.fish_key) python scripts/fish_credit.py)."""
from fishaudio import FishAudio

print(float(FishAudio().account.get_credits().credit))
