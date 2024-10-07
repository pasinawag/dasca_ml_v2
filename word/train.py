import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Function to pad or truncate sequences to a fixed length
def pad_or_truncate(data, max_length):
    padded_data = []
    for sequence in data:
        # If the sequence is shorter, pad with zeros
        if len(sequence) < max_length:
            padded_sequence = sequence + [0] * (max_length - len(sequence))
        # If the sequence is longer, truncate it
        else:
            padded_sequence = sequence[:max_length]
        padded_data.append(padded_sequence)
    return np.array(padded_data)


# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Find the length of the longest sequence
max_length = max(len(seq) for seq in data)

# Pad or truncate all sequences to have the same length
data_padded = pad_or_truncate(data, max_length)

# Convert to NumPy arrays
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
