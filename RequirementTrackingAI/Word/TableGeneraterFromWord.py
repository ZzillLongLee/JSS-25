import re
import pandas as pd
from win32com.client import Dispatch


def extract_text_from_word(file_path):
    # Open the Word application
    word = Dispatch("Word.Application")
    word.Visible = False  # Make Word invisible during the process
    doc = word.Documents.Open(file_path)

    # Read all text from the document
    full_text = ""
    for paragraph in doc.Paragraphs:
        full_text += paragraph.Range.Text + "\n"

    doc.Close(False)  # Close the document without saving
    word.Quit()  # Quit Word application
    return full_text


def extract_and_convert_multiple_questions(text):
    # Define regex pattern for extracting all questions
    pattern = r"문장\s*1:\s*(.*?)\s*\((SSS|SSDD|SSRS|SRS|HRS|SDD|HDD)\).*?문장\s*2:\s*(.*?)\s*\((SSS|SSDD|SSRS|SRS|HRS|SDD|HDD)\)"

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Create a table (dataframe) with the extracted data
        data = []
        for match in matches:
            sentence1, sentence1_Type, sentence2, sentence2_Type = match
            data.append([sentence1, sentence2, sentence1_Type.strip(), sentence2_Type.strip()])

        df = pd.DataFrame(data, columns=["문장1", "문장2", "문장1 Type", "문장2 Type"])
        return df
    else:
        print("No matches found in the text!")
        return None


# File path to the Word document
file_path = r"C:\Users\DTaQ\Desktop\questionnaire V2.0.docx"

# Extract text from the Word document
text = extract_text_from_word(file_path)

# Extract text and convert to table
table = extract_and_convert_multiple_questions(text)
if table is not None:
    print(table)
    # Save table to a CSV file
    table.to_csv("output_table.csv", index=False, encoding="utf-8-sig")
