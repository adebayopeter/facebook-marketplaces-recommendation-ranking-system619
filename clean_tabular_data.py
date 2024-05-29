# Import libraries
import pandas as pd
import csv

# Load dataset
df_images = pd.read_csv('data/csv/Images.csv')
df_products = pd.read_csv('data/csv/Products.csv',
                          lineterminator='\n',
                          quoting=csv.QUOTE_ALL)


# Display the first few rows of the dataframe
# print(df_images.head())
# print(df_images.columns)

# Removed all rows with any null data
df_cleaned_products = df_products.dropna()
df_cleaned_images = df_images.dropna()

# Convert prices to numerica values
df_cleaned_products['price'] = df_cleaned_products['price'].str.replace('£', '')
df_cleaned_products['price'] = df_cleaned_products['price'].str.replace(',', '')
df_cleaned_products['price'] = pd.to_numeric(df_cleaned_products['price'])

# Save cleaned data to csv
df_cleaned_products.to_csv('data/cleaned_products.csv', index=False)
df_cleaned_images.to_csv('data/cleaned_images.csv', index=False)

# print(df_products.head())
# print(df_products.columns)

##############
# Task 2
##############
# Load cleaned dataset
df_cleaned_products = pd.read_csv('data/csv/cleaned_products.csv',
                                  lineterminator='\n',
                                  quoting=csv.QUOTE_ALL)
df_cleaned_images = pd.read_csv('data/csv/cleaned_images.csv')

# Extract the root category from the category column
df_cleaned_products['root_category'] = (
    df_cleaned_products['category'].apply(lambda x: x.split('/')[0].strip()))

# Create encoder for root categories
unique_categories = df_cleaned_products['root_category'].unique()
encoder = {category: idx for idx, category in enumerate(unique_categories)}
decoder = {idx: category for category, idx in encoder.items()}

# Map root category labels to products
df_cleaned_products['category_label'] = df_cleaned_products['root_category'].map(encoder)

# Merge Image dataset and product dataset to get category label for each image
df_merged_data = df_cleaned_images.merge(
    df_cleaned_products[['id', 'category_label']],
    left_on='product_id',
    right_on='id',
    how='left'
)

# Drop unnecessary columns and keep only image_id and category
df_training_data = (
    df_merged_data[['id_x', 'category_label']].rename(columns={'id_x': 'image_id'}))

df_merged_data.to_csv('data/merged_data.csv')
df_training_data.to_csv('data/training_data.csv')

# print(encoder)
# print(decoder)


# print(df_cleaned_products['category'].head())
# print(df_cleaned_products.columns)



