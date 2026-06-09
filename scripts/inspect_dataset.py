import pandas as pd

df = pd.read_csv("data/plantseg/Metadata.csv")

print(f"Images: {len(df)}")
print(f"Plants: {df['Plant'].nunique()}")
print(f"Diseases: {df['Disease'].nunique()}")

print("\nTop 20 diseases:")
print(df["Disease"].value_counts().head(20))
