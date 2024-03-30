

def prepare_sequences(data):
    
    sequences = []
    labels = []
    
    for user_data in data.groupby('user_id'):
        
        user_df = user_data[1]
        temp_sequence = []
        count = 0
        
        for index, row in user_df.tail(100).iterrows():
            
            if count == len(user_df) - 1 or count == 99:

                label = [1] if row["has_disorder"] == True else [0]
                labels.append(label)
                
            else:
                temp_sequence.append(row['tweet_embedding'])
                count += 1
            
        sequences.append(temp_sequence)
        
    return sequences, labels