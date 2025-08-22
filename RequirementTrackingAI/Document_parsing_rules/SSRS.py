from word.DocxParser import DocxParser
from dao.Output import Output
import re

class SSRS_Document:

    def __init__(self, filePath, req_Start_Sec_Name, req_end_Sec_Name, outputType):
        self.keywords = ['식별자', '요구사항', '항목']
        self.ssrs_outputs = {}
        self.filePath = filePath
        self.req_Start_Sec_Name = req_Start_Sec_Name
        self.req_end_Sec_Name = req_end_Sec_Name
        self.outputType = outputType
        self.req_description_tables = None

        self.__getTables()

    def __getTables(self):
        docxParser = DocxParser(self.filePath)
        req_description_tables = docxParser.extract_tables_between_titles(self.req_Start_Sec_Name,
                                                                          self.req_end_Sec_Name)
        self.__generateSSRS_Output(req_description_tables)

    # Function to normalize column names by removing all kinds of whitespace
    def __normalize_column_name(self, col_name):
        return re.sub(r'\s+', '', col_name)

    def __generateSSRS_Output(self, req_description_tables):
        for table in req_description_tables:
            transposed_df = table.transpose()
            # Rename columns using the values in the first row
            transposed_df.columns = transposed_df.iloc[0]
            preprocessed_df = transposed_df.drop(0)
            # Normalize column names by removing all kinds of whitespace
            preprocessed_df.columns = [self.__normalize_column_name(col) for col in preprocessed_df.columns]

            # Filter columns based on keywords
            keyword_columns = {keyword: [col for col in transposed_df.columns if keyword in col] for keyword in self.keywords}

            # Check if all keywords are present
            if all(keyword_columns[keyword] for keyword in self.keywords):
                id = preprocessed_df.iloc[0][self.keywords[0]]
                category = preprocessed_df.iloc[0][self.keywords[1]]
                description = preprocessed_df.iloc[0][self.keywords[2]]
                ssrsOutput = Output(id, description, self.outputType)
                ssrsOutput.setCategory(category)
                self.ssrs_outputs[id] = ssrsOutput

    def getSSRS_outputs(self):
        return self.ssrs_outputs