import requests
import shutil
import os
import re
import datetime as dt
import sys
import pandas as pd
import zipfile as zf

# Downloading and Unpacking data

url_base = 'http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/'
enron_list = ["enron1", "enron2", "enron3", "enron4", "enron5", "enron6"]

if not os.path.exists("raw data"):
    os.mkdir("raw data")

for entry in enron_list:
    print("Downloading archive: " + entry + "...")
    # Download current enron archive
    url = url_base + entry + ".tar.gz"
    r = requests.get(url)
    path = os.path.join("raw data", f"{entry}.tar.gz")

    with open(path, 'wb') as f:
        f.write(r.content)
    print('...done! Archive saved to:', path)

    # Unpack current archive; this will unpack to /raw data/enron1, etc
    print("Unpacking contents of " + entry + " archive...")
    shutil.unpack_archive(path, "raw data/")
    print("...done! Archive unpacked to: raw data/" + entry)

print("All email archives downloaded and unpacked. Now beginning processing of email text files.")

# Processing data
# The data is recorded in such a way that each message is in a separate file.
# Therefore, we have to open each single file, parse it and add it to a dataframe

mails_list = []

print("Processing directories...")
# go through all dirs in the list
# each dir contains a ham & spam folder
for directory in enron_list:
    print("...processing " + str(directory) + "...")
    ham_folder = os.path.join("raw data", directory, "ham")
    spam_folder = os.path.join("raw data", directory, "spam")

    # Process ham messages in directory
    for file_entry in os.scandir(ham_folder):
        # This should be encoded in Latin_1, but catch encoding errors just to be sure
        try:
            with open(file_entry, 'r', encoding="latin_1") as file:
                content = file.read().split("\n", 1)
        except UnicodeDecodeError as e:
            print(f"COULD NOT DECODE: {e}")
            print("Problem with file:", file_entry)
            continue

        subject = content[0].replace("Subject: ", "")
        message = content[1]
        # date is contained in filename - parsed using regex pattern
        pattern = r"\d+\.(\d+-\d+-\d+)"
        date = re.search(pattern, str(file_entry)).group(1)
        date = dt.datetime.strptime(date, '%Y-%m-%d')
        mails_list.append([subject, message, "ham", date])

    # Process spam messages in directory
    for file_entry in os.scandir(spam_folder):
        try:
            with open(file_entry, 'r', encoding="latin_1") as file:
                content = file.read().split("\n", 1)
        except UnicodeDecodeError as e:
            print(f"COULD NOT DECODE: {e}")
            print("Problem with file:", file_entry)
            continue

        subject = content[0].replace("Subject: ", "")
        message = content[1]
        # date is contained in filename - parsed using regex pattern
        pattern = r"\d+\.(\d+-\d+-\d+)"
        date = re.search(pattern, str(file_entry)).group(1)
        date = dt.datetime.strptime(date, '%Y-%m-%d')
        mails_list.append([subject, message, "spam", date])

    print(str(directory)+" processed!")

print("All directories processed. Writing to DataFrame...")
mails = pd.DataFrame(mails_list, columns=["Subject", "Message", "Spam/Ham", "Date"])
print("...done!")

# Save to file
print("Saving data to file...")
with zf.ZipFile('enron_spam_data.zip', 'w') as enron_zip:
    enron_zip.writestr('enron_spam_data.csv', mails.to_csv(index_label = "Message ID", sep=',', escapechar='\\'), compress_type=zf.ZIP_DEFLATED)
print("...done! Compressed data saved to 'enron_spam_data.zip'")

# Confirmation message and data count
print("\nData processed and saved to file.\nMails contained in data:")
print("\nTotal:\t" + str(mails.shape[0]))
print(mails["Spam/Ham"].value_counts())
