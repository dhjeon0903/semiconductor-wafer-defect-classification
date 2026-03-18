import pandas as pd

print("Loading old pickle...")
df = pd.read_pickle("wm811k.pkl")

print("Loaded shape:", df.shape)
print("Saving converted pickle...")
df.to_pickle("wm811k_converted.pkl")

print("Done.")
print("Created: wm811k_converted.pkl")