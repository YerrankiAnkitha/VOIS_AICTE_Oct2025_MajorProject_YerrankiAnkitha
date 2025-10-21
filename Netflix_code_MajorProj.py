import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- 1. Data Loading and Initial Cleaning ---

# Load the dataset
df = pd.read_csv('Netflix Dataset.csv')

print("--- Data Loading and Initial Inspection ---")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Rename columns for easier access (optional but good practice)
df.columns = df.columns.str.replace(' ', '_')

# Handle missing values
# For 'Country', we'll fill with 'Unknown' for now for distribution analysis
df['Country'] = df['Country'].fillna('Unknown')
# For 'Director' and 'Cast', they are not critical for the main objectives but good to fill.
df['Director'] = df['Director'].fillna('Unknown')
df['Cast'] = df['Cast'].fillna('Unknown')

# Extract Release Year
# Convert 'Release_Date' to datetime objects and extract the year
df['Release_Year'] = pd.to_datetime(df['Release_Date']).dt.year

print("\nData Pre-processing complete. 'Release_Year' extracted.")
print("-" * 50)

# --- 2. Analysis 1: Distribution of Movies vs. TV Shows Over the Years ---

# Filter out records before 2008 as they might be sparse or outliers (optional, based on problem statement snippet)
# The snippet suggests data from 2008 to 2021, so let's use the full range.
content_by_year = df.groupby(['Release_Year', 'Category']).size().unstack(fill_value=0)

# Calculate the proportion of Movies and TV Shows each year
content_by_year['Total'] = content_by_year.sum(axis=1)
content_by_year['Movie_Proportion'] = (content_by_year['Movie'] / content_by_year['Total']) * 100
content_by_year['TV_Show_Proportion'] = (content_by_year['TV Show'] / content_by_year['Total']) * 100

# Plot the total number of releases over the years
plt.figure(figsize=(12, 6))
sns.lineplot(data=content_by_year[['Movie', 'TV Show']])
plt.title('Total Number of Movies vs. TV Shows Released Over Years', fontsize=16)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.grid(True)
plt.show()

# Plot the proportion of Movies vs. TV Shows each year (Stacked Area Chart for better comparison)
content_by_year[['Movie', 'TV Show']].plot(kind='area', stacked=True, figsize=(12, 6), alpha=0.7)
plt.title('Content Distribution by Type Over Years (Stacked)', fontsize=16)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.legend(title='Category')
plt.grid(axis='y')
plt.show()

print("--- Analysis 1 Complete: Movie vs. TV Show Distribution ---")
print(content_by_year[['Movie', 'TV Show', 'Total']].tail())
print("-" * 50)

# --- 3. Analysis 2: Most Common Genres and their Popularity Change ---

# The 'Type' column contains multiple genres separated by commas. We need to split and count.

def count_genres(dataframe, column_name):
    """Splits and counts all genres in a column."""
    genre_list = []
    # Drop missing values first
    df_temp = dataframe.dropna(subset=[column_name]).copy()
    # Split the string, clean up whitespace, and extend the list
    for entry in df_temp[column_name]:
        genres = [genre.strip() for genre in entry.split(',')]
        genre_list.extend(genres)
    return Counter(genre_list)

# Get overall top 10 genres
genre_counts = count_genres(df, 'Type')
top_10_genres = pd.DataFrame(genre_counts.most_common(10), columns=['Genre', 'Count'])

print("Top 10 Most Common Genres (Overall):")
print(top_10_genres)

# Plot overall top 10 genres
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Genre', data=top_10_genres, palette='viridis')
plt.title('Top 10 Most Common Genres on Netflix (Overall)', fontsize=16)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.show()

# Genre popularity change over time (Focus on top 5 for clarity in the time series plot)
top_5_genres_names = [item[0] for item in genre_counts.most_common(5)]
genre_over_time = []

# Iterate over each row and year
for index, row in df.iterrows():
    if pd.notna(row['Type']) and pd.notna(row['Release_Year']):
        genres = [g.strip() for g in row['Type'].split(',')]
        for genre in genres:
            if genre in top_5_genres_names:
                genre_over_time.append({'Release_Year': row['Release_Year'], 'Genre': genre})

genre_trend_df = pd.DataFrame(genre_over_time)
genre_yearly_count = genre_trend_df.groupby(['Release_Year', 'Genre']).size().unstack(fill_value=0)

# Plot top 5 genre trends over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=genre_yearly_count)
plt.title('Top 5 Genre Popularity Over Release Years', fontsize=16)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Count of Titles', fontsize=12)
plt.legend(title='Genre', loc='upper left')
plt.grid(True)
plt.show()

print("--- Analysis 2 Complete: Genre Trends ---")
print("-" * 50)

# --- 4. Analysis 3: Country-wise Contributions to Netflix's Catalog ---

# The 'Country' column may also contain multiple countries. We need to split and count.
country_counts = count_genres(df, 'Country')
top_10_countries = pd.DataFrame(country_counts.most_common(11), columns=['Country', 'Count'])
# Remove 'Unknown' if it appears in the top results
top_10_countries = top_10_countries[top_10_countries['Country'] != 'Unknown'].head(10)

print("Top 10 Contributing Countries (excluding Unknown):")
print(top_10_countries)

# Plot country contributions
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Country', data=top_10_countries, palette='rocket')
plt.title('Top 10 Countries by Content Contribution to Netflix', fontsize=16)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()

# Proportion of titles from the top country vs. others
top_country_name = top_10_countries.iloc[0]['Country']
top_country_count = top_10_countries.iloc[0]['Count']
other_countries_count = df.shape[0] - top_country_count - country_counts.get('Unknown', 0)

country_pie_data = pd.Series([top_country_count, other_countries_count, country_counts.get('Unknown', 0)],
                             index=[top_country_name, 'Other Countries', 'Unknown Country'])

plt.figure(figsize=(8, 8))
plt.pie(country_pie_data, labels=country_pie_data.index, autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette('pastel'), explode=[0.1, 0, 0])
plt.title('Content Distribution by Top Country vs. Others', fontsize=16)
plt.show()

print("--- Analysis 3 Complete: Country Contributions ---")
print("-" * 50)

print("\n--- Project Analysis Complete ---")
print("Visualizations and analysis steps have been generated as requested.")