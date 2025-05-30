import json
import random

def split_data(input_file, train_file, dev_file, split_ratio=0.9):
    """
    Splits a JSONL file into training and development sets.

    Args:
        input_file (str): Path to the input JSONL file.
        train_file (str): Path to save the training data.
        dev_file (str): Path to save the development data.
        split_ratio (float): Ratio of data to be used for training
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    random.shuffle(data)  # Shuffle the data before splitting

    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    dev_data = data[split_index:]

    with open(train_file, 'w', encoding='utf-8') as f_train:
        for item in train_data:
            f_train.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(dev_file, 'w', encoding='utf-8') as f_dev:
        for item in dev_data:
            f_dev.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Data split complete.")
    print(f"Total examples: {len(data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Development examples: {len(dev_data)}")

if __name__ == "__main__":
    input_filename = "./sharegpt_data/haruhi_sharegpt.jsonl"
    train_filename = "./sharegpt_data/haruhi_train.jsonl"
    dev_filename = "./sharegpt_data/haruhi_dev.jsonl"
    
    # Create a dummy haruhi_sharegpt.jsonl if it doesn't exist for testing
    try:
        with open(input_filename, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"'{input_filename}' not found. Creating a dummy file for demonstration.")
        dummy_data = [{"id": i, "text": f"This is sample text {i}"} for i in range(100)]
        with open(input_filename, 'w', encoding='utf-8') as f:
            for item in dummy_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Dummy '{input_filename}' created with 100 lines.")


    split_data(input_filename, train_filename, dev_filename)