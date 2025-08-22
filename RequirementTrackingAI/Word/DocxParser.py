import win32com.client
import hashlib
import pandas as pd


class DocxParser:

    def __init__(self, file_path):
        # Initialize the Word application
        self.word_app = win32com.client.Dispatch("Word.Application")
        self.word_app.Visible = False  # Keep Word application invisible
        try:
            # Open the Word document
            self.doc = self.word_app.Documents.Open(file_path)
        except Exception as e:
            print(f"An error occurred: {e}")

    def close_docx(self):
        # Close the document without saving changes
        self.doc.Close(SaveChanges=False)
        # Ensure Word application is properly closed
        self.word_app.Quit()

    def get_table_hash(self, table):
        """Generate a unique hash for a table based on its content."""
        table_content = ""
        for row_idx in range(1, table.Rows.Count + 1):
            for col_idx in range(1, table.Columns.Count + 1):
                try:
                    cell = table.Cell(Row=row_idx, Column=col_idx)
                    cell_text = cell.Range.Text.strip(chr(7)).strip()
                    table_content += cell_text
                except Exception as e:
                    # Handle the error for merged cells
                    table_content += ""  # Append an empty string or a placeholder
                    print(f"Error accessing cell at Row {row_idx}, Column {col_idx}: {e}")
        return hashlib.md5(table_content.encode('utf-8')).hexdigest()

    def extract_tables_between_titles(self, start_title, end_title):

        tables = []
        start_title_found = False
        end_title_found = False
        processed_table_hashes = set()  # Track processed table hashes to avoid duplication

        # Iterate through paragraphs to find the start title and collect tables until the end title
        for para in self.doc.Paragraphs:

            para_text = para.Range.Text.strip()

            if start_title_found and para_text == end_title.strip():
                end_title_found = True
                break

            if start_title_found:
                # Check if the current paragraph has tables
                for table in para.Range.Tables:
                    table_hash = self.get_table_hash(table)  # Generate a unique hash for the table
                    if table_hash not in processed_table_hashes:
                        processed_table_hashes.add(table_hash)  # Mark this table as processed
                        table_data = []
                        for row_idx in range(1, table.Rows.Count + 1):
                            row_data = []
                            for col_idx in range(1, table.Columns.Count + 1):
                                try:
                                    cell = table.Cell(Row=row_idx, Column=col_idx)
                                    cell_text = cell.Range.Text.strip(chr(7)).strip()
                                    row_data.append(cell_text)
                                except Exception as e:
                                    # Handle the error for merged cells
                                    if len(table_data) > 0:
                                        prev_row_data = table_data[-1]
                                        prev_cell_data = prev_row_data[col_idx - 1]
                                        row_data.append(prev_cell_data)  # Append an empty string or a placeholder
                                    else:
                                        row_data.append('')
                                    print(f"Error accessing cell at Row {row_idx}, Column {col_idx}: {e}")
                            table_data.append(row_data)
                        df = pd.DataFrame(table_data)
                        tables.append(df)

            if para_text == start_title.strip():
                start_title_found = True

        if not start_title_found:
            raise ValueError(f"Start title '{start_title}' not found.")
        if start_title_found and not end_title_found:
            raise ValueError(f"End title '{end_title}' not found after the start title.")

        return tables

    def extract_table(self, table):
        table_data = []
        for row_idx in range(1, table.Rows.Count + 1):
            row_data = []
            for col_idx in range(1, table.Columns.Count + 1):
                try:
                    cell = table.Cell(Row=row_idx, Column=col_idx)
                    cell_text = cell.Range.Text.strip(chr(7)).strip()
                    row_data.append(cell_text)
                except Exception as e:
                    # Handle the error for merged cells
                    if len(table_data) > 0:
                        prev_row_data = table_data[-1]
                        prev_cell_data = prev_row_data[col_idx - 1]
                        row_data.append(prev_cell_data)  # Append an empty string or a placeholder
                    else:
                        row_data.append('')
            table_data.append(row_data)
        return table_data

    def getDoc(self):
        return self.doc


if __name__ == '__main__':

    # Usage
    file_path = r'C:\Users\DTaQ\Desktop\My task\2024 personal project\2024 연구그룹 과제\HRS\HW 요구사항 명세서_기만기검사기.docx'
    docxParser = DocxParser(file_path)
    tables = docxParser.extract_tables_between_titles('요구사항 추적성', '참고사항')

    # Print tables
    print("\nTables:")
    for table in tables:
        for row in table:
            print("\t".join(row))
        print()  # Blank line between tables

    docxParser.close_docx()
