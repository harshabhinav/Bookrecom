import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data
books = pd.read_csv('books.csv')  # Replace with actual dataset path
ratings = pd.read_csv('ratings.csv')
users = pd.read_csv('users.csv')

# Filter users with at least 200 ratings
user_counts = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 200].index)]

# Filter books with at least 100 ratings
book_counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 100].index)]

# Create a user-book matrix
book_pivot = ratings.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)

# Convert to sparse matrix
book_sparse = csr_matrix(book_pivot)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(book_sparse)

# Recommendation function
def get_recommends(book_title):
    if book_title not in book_pivot.index:
        return "Book not found"
    
    book_index = book_pivot.index.get_loc(book_title)
    distances, indices = knn.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)
    
    recommended_books = [[book_pivot.index[i], distances.flatten()[j]] for j, i in enumerate(indices.flatten()[1:])]
    return [book_title, recommended_books]

# Test
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
