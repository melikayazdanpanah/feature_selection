import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def relief(X, y, num_iterations=100):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    
    for _ in range(num_iterations):
        # Step 2: Randomly select an instance
        random_index = np.random.randint(num_samples)
        instance = X[random_index]
        
        # Step 3: Find nearest hit and nearest miss
        hit_index = None
        miss_index = None
        min_hit_distance = float('inf')
        min_miss_distance = float('inf')
        
        for i in range(num_samples):
            if i != random_index:
                if y[i] == y[random_index]:
                    distance = np.linalg.norm(X[i] - instance)
                    if distance < min_hit_distance:
                        hit_index = i
                        min_hit_distance = distance
                else:
                    distance = np.linalg.norm(X[i] - instance)
                    if distance < min_miss_distance:
                        miss_index = i
                        min_miss_distance = distance
        
        # Step 4: Update feature weights
        for j in range(num_features):
            weights[j] += (np.abs(X[random_index, j] - X[hit_index, j]) / num_iterations
                           - np.abs(X[random_index, j] - X[miss_index, j]) / num_iterations)
    
    return weights

# Load dataset from CSV file
seg_data = pd.read_csv("segmentation-all.csv") 
print(seg_data.shape)
seg_data.head()

# Extract target variable and features
y = seg_data.pop('Class').values
X_raw = seg_data.values

# Split dataset into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, 
                                                             random_state=42, test_size=1/2)

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Run Relief algorithm
feature_weights = relief(X_train, y_train)
print("Feature weights:", feature_weights)
