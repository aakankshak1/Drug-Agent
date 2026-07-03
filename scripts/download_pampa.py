from tdc.single_pred import ADME

data = ADME(name="PAMPA_NCATS")

df = data.get_data()

df.to_csv("data/PAMPA.csv", index=False)

print(df.head())