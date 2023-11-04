import pandas as pd

ratings = pd.read_csv('./datasets/Ratings.csv')
books   = pd.read_csv('./datasets/Books.csv')

books = books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)

merged_data = pd.merge(ratings, books, on='ISBN', how='left')

merged_data.to_csv('./datasets/joined.csv', index=False)