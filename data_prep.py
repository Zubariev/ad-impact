import pandas as pd
import calendar
from datetime import datetime, timedelta

# === CONFIG ===
CLIENT_BRAND = "Metro"
OOH_PATH = "MCO_Out-of-Home_16-07-2025.xlsx"
FOOT_TRAFFIC_PATH = "visits_diff_upd.xlsx"
OUTPUT_PATH = "dataset_3.csv"

# === HELPER FUNCTIONS ===
def get_week_to_month_mapping(year):
    """
    Create mapping of calendar weeks to months for a given year.
    Week is assigned to the month that contains most days of that week.
    """
    week_to_month = {}
    
    # Start from January 1st
    jan_1 = datetime(year, 1, 1)
    
    # Find the first Monday (start of first week)
    days_to_monday = (7 - jan_1.weekday()) % 7
    first_monday = jan_1 + timedelta(days=days_to_monday)
    if days_to_monday > 3:  # If Jan 1 is Thu-Sun, week 1 starts next Monday
        first_monday = jan_1 + timedelta(days=days_to_monday)
        week_num = 1
    else:  # If Jan 1 is Mon-Wed, week 1 already started
        first_monday = jan_1 - timedelta(days=jan_1.weekday())
        week_num = 1
    
    current_date = first_monday
    
    for week in range(1, 54):  # Max 53 weeks in a year
        if current_date.year > year:
            break
            
        week_end = current_date + timedelta(days=6)
        
        # Count days in each month for this week
        month_days = {}
        for day_offset in range(7):
            day = current_date + timedelta(days=day_offset)
            if day.year == year:
                month = day.month
                month_days[month] = month_days.get(month, 0) + 1
        
        # Assign week to month with most days
        if month_days:
            assigned_month = max(month_days, key=month_days.get)
            week_to_month[week] = assigned_month
        
        current_date += timedelta(days=7)
    
    return week_to_month

def get_weeks_in_month(year, month):
    """Get list of week numbers that belong to a specific month."""
    week_to_month = get_week_to_month_mapping(year)
    return [week for week, month_assigned in week_to_month.items() if month_assigned == month]

def convert_monthly_to_weekly(monthly_df, year):
    """
    Convert monthly aggregated data to weekly by distributing values proportionally.
    
    Args:
        monthly_df: DataFrame with monthly data (must have 'Month' column)
        year: Year for week mapping
    
    Returns:
        DataFrame with weekly data (Week column instead of Month)
    """
    weekly_records = []
    
    for _, row in monthly_df.iterrows():
        month = row['Month']
        weeks_in_month = get_weeks_in_month(year, month)
        
        if not weeks_in_month:
            continue
            
        # Number of weeks to distribute across
        num_weeks = len(weeks_in_month)
        
        # Create a record for each week
        for week in weeks_in_month:
            weekly_row = row.copy()
            
            # Replace Month with Week
            weekly_row = weekly_row.drop('Month')
            weekly_row['Week'] = week
            
            # Distribute numeric values proportionally
            for col in weekly_row.index:
                if col not in ['City', 'Week'] and pd.api.types.is_numeric_dtype(type(weekly_row[col])):
                    weekly_row[col] = weekly_row[col] / num_weeks
            
            weekly_records.append(weekly_row)
    
    if not weekly_records:
        return pd.DataFrame()
        
    weekly_df = pd.DataFrame(weekly_records)
    return weekly_df

# === Load Data ===
ooh_df = pd.read_excel(OOH_PATH)
foot_df = pd.read_excel(FOOT_TRAFFIC_PATH)

# === Clean Columns ===
ooh_df.columns = ooh_df.columns.str.strip()
foot_df.columns = foot_df.columns.str.strip()

# === Rename Columns ===
ooh_df = ooh_df.rename(columns={"Brands": "Brand", "Budget_Net_UAH": "Ad_Budget_UAH", "Number_ins": "Insertions"})
foot_df = foot_df.rename(columns={"city": "City", "visit": "Visits", "sales": "Sales_UAH", "promo_sales": "Promo_Sales_UAH"})

# === Replace City Names ===
city_map = {"Вінниця": "Vinnytsia", "Дніпро": "Dnipro", "Запоріжжя": "Zaporizhzhia", "Івано-Франківськ": "Ivano-Frankivsk", "Київ": "Kyiv", "Кривий Ріг": "Kryvyi Rig", "Луцьк": "Lutsk", "Миколаїв": "Mykolaiv", "Одеса": "Odesa", "Полтава": "Poltava", "Рівне": "Rivne", "Тернопіль": "Ternopil", "Харків": "Kharkiv", "Чернівці": "Chernivtsi", "Чернігів": "Chernihiv"}
ooh_df["City"] = ooh_df["City"].replace(city_map)
foot_df["City"] = foot_df["City"].replace(city_map)

# === Process Foot Traffic (Already Weekly) ===
# Parse week_id format (e.g., 202301 = 2023 week 1)
foot_df['Week_ID_Str'] = foot_df['week_id'].astype(str)
foot_df['Year'] = foot_df['Week_ID_Str'].str[:4].astype(int)
foot_df['Week'] = foot_df['Week_ID_Str'].str[4:].astype(int)

# Use foot traffic data as is (already weekly)
foot_weekly = foot_df[['City', 'Year', 'Week', 'Visits', 'Sales_UAH', 'Promo_Sales_UAH']].copy()

# === Prepare Client OOH (Convert Monthly to Weekly) ===
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

# Convert to weekly (assuming 2023 as the primary year - adjust as needed)
years_in_data = foot_weekly['Year'].unique()
client_weekly_data = []
for year in years_in_data:
    client_weekly_year = convert_monthly_to_weekly(client_monthly, year)
    if not client_weekly_year.empty:
        client_weekly_year['Year'] = year
        client_weekly_data.append(client_weekly_year)

client_weekly = pd.concat(client_weekly_data, ignore_index=True) if client_weekly_data else pd.DataFrame()

# === Prepare Competitor OOH (Convert Monthly to Weekly) ===
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

# Convert to weekly
competitor_weekly_data = []
for year in years_in_data:
    competitor_weekly_year = convert_monthly_to_weekly(competitor_monthly, year)
    if not competitor_weekly_year.empty:
        competitor_weekly_year['Year'] = year
        competitor_weekly_data.append(competitor_weekly_year)

competitor_weekly = pd.concat(competitor_weekly_data, ignore_index=True) if competitor_weekly_data else pd.DataFrame()

# === Prepare Individual Brand OOH (Convert Monthly to Weekly) ===
# Get all competitor brands (excluding client brand)
competitor_brands = ooh_df[ooh_df['Brand'] != CLIENT_BRAND]['Brand'].unique()
print(f"Processing individual metrics for brands: {list(competitor_brands)}")

# Create individual brand metrics
brand_weekly_data = []
for brand in competitor_brands:
    brand_monthly = ooh_df[ooh_df['Brand'] == brand].groupby(['City', 'Month']).agg({
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
    
    # Convert to weekly for each year
    brand_all_years = []
    for year in years_in_data:
        brand_weekly_year = convert_monthly_to_weekly(brand_monthly, year)
        if not brand_weekly_year.empty:
            brand_weekly_year['Year'] = year
            brand_all_years.append(brand_weekly_year)
    
    if brand_all_years:
        brand_weekly_combined = pd.concat(brand_all_years, ignore_index=True)
        brand_weekly_data.append(brand_weekly_combined)

# === Filter Cities ===
cities_with_visits = foot_weekly['City'].unique()
if not client_weekly.empty:
    client_weekly = client_weekly[client_weekly['City'].isin(cities_with_visits)]
if not competitor_weekly.empty:
    competitor_weekly = competitor_weekly[competitor_weekly['City'].isin(cities_with_visits)]

# Filter brand data for cities with visits
for i, brand_data in enumerate(brand_weekly_data):
    brand_weekly_data[i] = brand_data[brand_data['City'].isin(cities_with_visits)]

# === Merge ===
# Start with foot traffic data
df = foot_weekly.copy()

# Merge client data
if not client_weekly.empty:
    df = df.merge(client_weekly, on=['City', 'Year', 'Week'], how='left')

# Merge aggregated competitor data
if not competitor_weekly.empty:
    df = df.merge(competitor_weekly, on=['City', 'Year', 'Week'], how='left')

# Merge individual brand data
for brand_data in brand_weekly_data:
    if not brand_data.empty:
        df = df.merge(brand_data, on=['City', 'Year', 'Week'], how='left')

# === Fill NaNs ===
# Get all columns that need to be filled with 0
fill_cols = ['Client_GRP', 'Client_OTS', 'Client_Ad_Budget_UAH', 'Client_Ad_Events', 'Client_Insertions', 
             'Competitor_GRP', 'Competitor_OTS', 'Competitor_Ad_Budget_UAH', 'Competitor_Ad_Events', 'Competitor_Insertions']

# Add individual brand columns to fill list
for brand in competitor_brands:
    fill_cols.extend([f'{brand}_GRP', f'{brand}_OTS', f'{brand}_Ad_Budget_UAH', f'{brand}_Insertions'])

# Only fill columns that exist in the dataframe
existing_fill_cols = [col for col in fill_cols if col in df.columns]
df[existing_fill_cols] = df[existing_fill_cols].fillna(0)

# === Sort and Save ===
df = df.sort_values(['City', 'Year', 'Week']).reset_index(drop=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f" Complete dataset saved to {OUTPUT_PATH}")
print("Sample data:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Available brands in OOH data: {list(ooh_df['Brand'].value_counts().head().index)}")
print(f"Client brand '{CLIENT_BRAND}' found: {'Yes' if CLIENT_BRAND in ooh_df['Brand'].values else 'No'}")
print(f"Years processed: {sorted(years_in_data)}")
print(f"Weeks range: {df['Week'].min()}-{df['Week'].max()}")
