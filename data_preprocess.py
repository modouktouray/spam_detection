import pandas as pd
import zipfile as zf
import os
import csv

def read_data():

    file_path = "enron_spam_data.zip" 

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    print("\nReading in local data...", end="")
    mails = pd.read_csv(file_path, compression="zip", index_col="Message ID")
    print("Done!")

    return mails

def separate_training_testing_data(mails):
    """
    Separate data into training and testing datasets.
    """
    print("\nSeparating Data into training and testing data...", end="")
    # Randomize dataset
    mails = mails.sample(frac=1, random_state=42)

    # Reindex after randomization
    mails.reset_index(inplace=True, drop=True)

    # Get 80% as training data, rest as test data
    cutoff_index = int(round(mails.shape[0] * 0.8, 0))

    train = mails.iloc[:cutoff_index].copy(deep=True)
    test = mails.iloc[cutoff_index:].copy(deep=True)

    # mails are no longer needed - drop from memory
    del mails

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    print("Done!")
    print("\nCharacteristics of Training & Testing Data Set:")
    print("TRAINING DATA:")
    print("Proportion in %")
    print(round(train["Spam/Ham"].value_counts(normalize=True), 4) * 100)

    print("\nTESTING DATA:")
    print("Proportion in %")
    print(round(test["Spam/Ham"].value_counts(normalize=True), 4) * 100)

    return train, test

def save_to_zip(data, file_name):
    """
    Save DataFrame to a zip file.
    """
    print(f"\nSaving {file_name} data set and dropping it from memory...", end="")
    with zf.ZipFile(f'data/{file_name}.zip', 'w') as data_zip:
        data_zip.writestr(f'{file_name}.csv', data.to_csv(index_label="Message ID"), compress_type=zf.ZIP_DEFLATED)
    print("DONE!\nDataFrame saved to 'data/{file_name}.zip'.")
    del data_zip, data

def save_dict_to_csv(dictionary, output_file_name, progress_step_size=5):
    """
    Save a dictionary to a CSV file.
    """
    print(f"...saving {output_file_name} data to csv-file...")
    with open(output_file_name, 'w', newline='') as csvfile:
        print("...writing dictionary to file...")
        writer = csv.writer(csvfile)
        writer.writerow(dictionary)  # First row (the keys of the dictionary).
        print("...wrote header to file...")
        print("...now writing values to file...")

        dict_vals_zip = zip(*dictionary.values())

        # vals for progress messages
        steps = len(list(dict_vals_zip))
        steps_mult = round(steps / 100)
        current_step = 0
        progress = 0

        for values in zip(*dictionary.values()):
            writer.writerow(values)

            current_step += 1

            if current_step % steps_mult == 0:
                progress += 1
                if progress % progress_step_size == 0:
                    print("...about " + str(progress) + "% done...", end="\r")

    print("...Done!                   ")

def save_to_zip(data, file_name):
    """
    Save DataFrame to a zip file.
    """
    save_dir = 'data'
    save_path = f'{save_dir}/{file_name}.zip'

    # Create the 'data' directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"\nSaving {file_name} data set to {save_path} and dropping it from memory...", end="")

    # Debugging: Print the current working directory
    print("Current Working Directory:", os.getcwd())

    with zf.ZipFile(save_path, 'w') as data_zip:
        data_zip.writestr(f'{file_name}.csv', data.to_csv(index_label="Message ID"), compress_type=zf.ZIP_DEFLATED)

    print("DONE!\nDataFrame saved to", save_path)
    del data_zip, data


def main():
    # download_enron_spam_data()  # Commenting out the download function
    mails = read_data()
    if mails is None:
        return

    train, test = separate_training_testing_data(mails)

    save_to_zip(train, 'train_data')
    save_to_zip(test, 'test_data')

    # Continue with the rest of your code...

if __name__ == "__main__":
    main()
