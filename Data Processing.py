import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

def load_data(dataset_path):
    """
    Load the dataset from the specified path.
    
    Args:
        dataset_path (str): Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        return pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError:
        print("The dataset file is empty.")
        return None
    except pd.errors.ParserError:
        print("There was an error parsing the dataset file.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {str(e)}")
        return None

def extract_primary_category(product_category_tree):
    """
    Extract the primary category from the product category tree.
    
    Args:
        product_category_tree (str): Product category tree string.
        
    Returns:
        str: Primary category if found, None otherwise.
    """
    try:
        categories = eval(product_category_tree)
        if isinstance(categories, list) and len(categories) > 0:
            return categories[0].split('>>')[0].strip()
    except (ValueError, SyntaxError, IndexError, TypeError):
        pass
    return None

def extract_primary_image(image_str):
    """
    Extract the primary image URL from the image string.
    
    Args:
        image_str (str): Image string containing image URLs.
        
    Returns:
        str: Primary image URL if found, None otherwise.
    """
    try:
        images = eval(image_str)
        if isinstance(images, list) and len(images) > 0:
            return images[0]
    except (ValueError, SyntaxError, TypeError):
        pass
    return None

def determine_gender(product_name, description):
    """
    Determine the gender based on the product name and description.
    
    Args:
        product_name (str): Product name.
        description (str): Product description.
        
    Returns:
        str: Gender category ('Women', 'Men', or 'Unisex').
    """
    keywords_women = ['women', 'woman', 'female', 'girls', 'girl', 'ladies', 'lady']
    keywords_men = ['men', 'man', 'male', 'boys', 'boy', 'gentlemen', 'gentleman']

    name_desc = f"{str(product_name).lower()} {str(description).lower()}"
    if any(keyword in name_desc for keyword in keywords_women):
        return 'Women'
    elif any(keyword in name_desc for keyword in keywords_men):
        return 'Men'
    else:
        return 'Unisex'

def preprocess_data(df, save_path):
    """
    Preprocess the dataset by extracting relevant information and cleaning the data.
    
    Args:
        df (pd.DataFrame): Input dataset DataFrame.
        save_path (str): Path to save the preprocessed dataset.
        
    Returns:
        pd.DataFrame: Preprocessed dataset DataFrame.
    """
    # Replace "No rating available" with a random number between 2 and 5
    df['rating'] = df['rating'].apply(lambda x: random.uniform(2, 5) if x == "No rating available" else x)

    # Extract primary categories, primary image links, and gender
    df['primary_category'] = df['product_category_tree'].apply(extract_primary_category)
    df['primary_image_link'] = df['image'].apply(extract_primary_image)
    df['gender'] = df.apply(lambda x: determine_gender(x['product_name'], x['description']), axis=1)

    # Calculate discount percentage
    try:
        df['discount_percentage'] = ((df['retail_price'] - df['discounted_price']) / df['retail_price']) * 100
    except KeyError as e:
        print(f"Missing column for discount calculation: {e}")
        df['discount_percentage'] = None

    # Select relevant columns
    columns_of_interest = ['pid', 'product_url', 'product_name', 'primary_category',
                           'retail_price', 'discounted_price', 'primary_image_link',
                           'description', 'brand', 'gender', 'discount_percentage', 'rating']
    refined_df = df[columns_of_interest]

    # Drop rows with missing critical data
    refined_df = refined_df.dropna(subset=['primary_category', 'retail_price', 'discounted_price'])

    # Save the preprocessed dataset
    refined_df.to_csv(save_path, index=False)
    print(f"Preprocessed dataset saved to {save_path}")

    return refined_df

def display_data_analysis(refined_df):
    """
    Display data analysis visualizations.
    
    Args:
        refined_df (pd.DataFrame): Preprocessed dataset DataFrame.
    """
    # Top categories analysis
    top_categories = refined_df['primary_category'].value_counts().nlargest(10).index
    top_categories_df = refined_df[refined_df['primary_category'].isin(top_categories)]

    # Boxplot for price distribution across top categories
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='retail_price', y='primary_category', data=top_categories_df, ax=ax)
    ax.set_title('Price Distribution Across Top Categories')
    ax.set_xlabel('Retail Price')
    ax.set_ylabel('Category')
    plt.show()

    # Histogram for discount percentage
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(refined_df['discount_percentage'], bins=30, kde=True, ax=ax)
    ax.set_title('Discount Percentage Distribution')
    ax.set_xlabel('Discount Percentage')
    ax.set_ylabel('Number of Products')
    plt.show()

# Example Usage
if __name__ == "__main__":
    dataset_path = "D:\\Projects\\Brilyant recomendation engine\\Dataset\\flipkart_com-ecommerce_sample.csv"
    save_path = "D:\\Projects\\Brilyant recomendation engine\\Dataset\\preprocessed_dataset.csv"
    df = load_data(dataset_path)

    if df is not None:
        refined_df = preprocess_data(df, save_path)
        display_data_analysis(refined_df)
