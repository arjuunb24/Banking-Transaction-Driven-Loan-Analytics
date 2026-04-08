import os

folders = [
    "data/raw",
    "data/processed",
    "data/outputs",
    "sql/views",
    "src",
    "notebooks"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

print("\nProject structure ready.")
print("Place PaySim CSV in: data/raw/paysim.csv")
print("Place Lending Club CSV in: data/raw/lending_club.csv")