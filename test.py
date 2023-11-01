import os
import pandas as pd
from argparse import ArgumentParser
from copyrightfpd.CopyrightFPD import *
from sklearn.metrics import classification_report

def main():
    parser = ArgumentParser(description="Read CSV file and check model file")
    parser.add_argument("--csv-file", help="Path to the CSV file - it should two columns: text and label")

    args = parser.parse_args()
    
    # Read the CSV file using Pandas
    try:
        data = pd.read_csv(args.csv_file_path)
        print(f"Successfully read the CSV file from: {args.csv_file_path}")
    except FileNotFoundError:
        print(f"CSV file not found at: {args.csv_file_path}")
        return
    
    # Check if the copyrightfpd package is installed in the fossy user pythondeps
    # Simply check if a directory containing copyrightfpd is inside /home/fossy/pythondeps
    dirs = os.listdir('/home/fossy/pythondeps')
    dirs = [d for d in dirs if 'copyrightfpd' in d]

    if len(dirs) == 0:
        print("""The copyrightfpd package is not installed in the fossy user pythondeps. 
                Please install by running the post-install script with the --python-experimental flag""")
        return
    
    # Create an instance of the class
    copyrightFPD = CopyrightFPD()
    
    predictions = copyrightFPD.predict(data['text'].to_list())

    # Print the predictions
    print(classification_report(data['label'].to_list(), predictions))

if __name__ == "__main__":
    main()
