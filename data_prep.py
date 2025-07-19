import pandas as pd

# === CONFIG ===
CLIENT_BRAND = "Metro"
OOH_PATH = "MCO_Out-of-Home_16-07-2025.xlsx"
FOOT_TRAFFIC_PATH = "visits_diff_upd.xlsx"
OUTPUT_PATH = "dataset_3.csv"

# === Load Data ===
ooh_df = pd.read_excel(OOH_PATH)
foot_df = pd.read_excel(FOOT_TRAFFIC_PATH)

# === Clean Columns ===
ooh_df.columns = ooh_df.columns.str.strip()
foot_df.columns = foot_df.columns.str.strip()

# === Rename Columns ===
ooh_df = ooh_df.rename(columns={"Brands": "Brand", "Budget_Net_UAH": "Ad_Budget_UAH", "Number_ins": "Insertions"})
foot_df = foot_df.rename(columns={"city": "City", "visit": "Visits", "sales": "Sales_UAH", "promo_sales": "Promo_Sales_UAH", "data_period": "Date"})

# === Replace City Names ===
city_map = {"Вінниця": "Vinnytsia", "Дніпро": "Dnipro", "Запоріжжя": "Zaporizhzhia", "Івано-Франківськ": "Ivano-Frankivsk", "Київ": "Kyiv", "Кривий Ріг": "Kryvyi Rig", "Луцьк": "Lutsk", "Миколаїв": "Mykolaiv", "Одеса": "Odesa", "Полтава": "Poltava", "Рівне": "Rivne", "Тернопіль": "Ternopil", "Харків": "Kharkiv", "Чернівці": "Chernivtsi", "Чернігів": "Chernihiv"}
ooh_df["City"] = ooh_df["City"].replace(city_map)
foot_df["City"] = foot_df["City"].replace(city_map)

# === Extract Month and Year ===
foot_df['Date'] = pd.to_datetime(foot_df['Date'], dayfirst=True, errors='coerce')
foot_df['Month'] = foot_df['Date'].dt.month
foot_df['Year'] = foot_df['Date'].dt.year

# === Aggregate Foot Traffic Monthly ===
foot_monthly = foot_df.groupby(['City', 'Year', 'Month']).agg({'Visits': 'sum', 'Sales_UAH': 'sum', 'Promo_Sales_UAH': 'sum'}).reset_index()

# === Prepare Client OOH ===
client_ooh = ooh_df[ooh_df['Brand'] == CLIENT_BRAND]
client_monthly = client_ooh.groupby(['City', 'Month']).agg({
    'GRP': 'sum', 
    'OTS': 'sum', 
    'Ad_Budget_UAH': 'sum', 
    'Ad events': 'count', 
    'Insertions': 'sum'
}).reset_index().rename(columns={
    'GRP': 'Client_GRP', 
    'OTS': 'Client_OTS', 
    'Ad_Budget_UAH': 'Client_Ad_Budget_UAH', 
    'Ad events': 'Client_Ad_Events', 
    'Insertions': 'Client_Insertions'
})

# === Prepare Competitor OOH ===
competitor_ooh = ooh_df[ooh_df['Brand'] != CLIENT_BRAND]
competitor_monthly = competitor_ooh.groupby(['City', 'Month']).agg({
    'GRP': 'sum', 
    'OTS': 'sum', 
    'Ad_Budget_UAH': 'sum', 
    'Ad events': 'count', 
    'Insertions': 'sum'
}).reset_index().rename(columns={
    'GRP': 'Competitor_GRP', 
    'OTS': 'Competitor_OTS', 
    'Ad_Budget_UAH': 'Competitor_Ad_Budget_UAH', 
    'Ad events': 'Competitor_Ad_Events', 
    'Insertions': 'Competitor_Insertions'
})

# === Prepare Individual Brand OOH ===
# Get all competitor brands (excluding client brand)
competitor_brands = ooh_df[ooh_df['Brand'] != CLIENT_BRAND]['Brand'].unique()
print(f"Processing individual metrics for brands: {list(competitor_brands)}")

# Create individual brand metrics
brand_monthly_data = []
for brand in competitor_brands:
    brand_data = ooh_df[ooh_df['Brand'] == brand].groupby(['City', 'Month']).agg({
        'GRP': 'sum', 
        'OTS': 'sum', 
        'Ad_Budget_UAH': 'sum', 
        'Insertions': 'sum'
    }).reset_index().rename(columns={
        'GRP': f'{brand}_GRP', 
        'OTS': f'{brand}_OTS', 
        'Ad_Budget_UAH': f'{brand}_Ad_Budget_UAH', 
        'Insertions': f'{brand}_Insertions'
    })
    brand_monthly_data.append(brand_data)

# === Filter Cities ===
cities_with_visits = foot_monthly['City'].unique()
client_monthly = client_monthly[client_monthly['City'].isin(cities_with_visits)]
competitor_monthly = competitor_monthly[competitor_monthly['City'].isin(cities_with_visits)]

# Filter brand data for cities with visits
for i, brand_data in enumerate(brand_monthly_data):
    brand_monthly_data[i] = brand_data[brand_data['City'].isin(cities_with_visits)]

# === Merge ===
# Start with foot traffic data
df = foot_monthly.copy()

# Merge client data
df = df.merge(client_monthly, on=['City', 'Month'], how='left')

# Merge aggregated competitor data
df = df.merge(competitor_monthly, on=['City', 'Month'], how='left')

# Merge individual brand data
for brand_data in brand_monthly_data:
    df = df.merge(brand_data, on=['City', 'Month'], how='left')

# === Fill NaNs ===
# Get all columns that need to be filled with 0
fill_cols = ['Client_GRP', 'Client_OTS', 'Client_Ad_Budget_UAH', 'Client_Ad_Events', 'Client_Insertions', 
             'Competitor_GRP', 'Competitor_OTS', 'Competitor_Ad_Budget_UAH', 'Competitor_Ad_Events', 'Competitor_Insertions']

# Add individual brand columns to fill list
for brand in competitor_brands:
    fill_cols.extend([f'{brand}_GRP', f'{brand}_OTS', f'{brand}_Ad_Budget_UAH', f'{brand}_Insertions'])

df[fill_cols] = df[fill_cols].fillna(0)

# === Sort and Save ===
df = df.sort_values(['City', 'Year', 'Month']).reset_index(drop=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Complete dataset saved to {OUTPUT_PATH}")
print("Sample data:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Available brands in OOH data: {list(ooh_df['Brand'].value_counts().head().index)}")
print(f"Client brand '{CLIENT_BRAND}' found: {'Yes' if CLIENT_BRAND in ooh_df['Brand'].values else 'No'}")
