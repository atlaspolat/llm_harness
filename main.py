from datasets import load_dataset

   
def main():

    dataset = load_dataset("AtlasPolat/yks2024", streaming=False)
    print("Dataset loaded successfully!")
    print(f"Dataset info: {dataset}")
        
    print("First few examples:")
    for i, example in enumerate(dataset['train']):
        if i < 5:  # Print only the first 5 examples
            print(f"Example {i+1}: {example}")
        else:
            break
   
if __name__ == "__main__":
    main()
