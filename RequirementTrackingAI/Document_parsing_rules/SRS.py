import pandas as pd
from word.DocxParser import DocxParser
import re
from dao.Output import Output


class SRS_Document:

    def __init__(self, filePath, req_Start_Sec_Name, req_end_Sec_Name, req_Trace_Start_Sec_Name,
                 req_Trace_End_Sec_Name, outputType):
        self.keywords = ['식별자', '요구사항']
        self.hdd_outputs = {}
        self.outputType = outputType
        self.filePath = filePath
        self.req_Start_Sec_Name = req_Start_Sec_Name
        self.req_end_Sec_Name = req_end_Sec_Name

        self.req_Trace_Start_Sec_Name = req_Trace_Start_Sec_Name
        self.req_Trace_End_Sec_Name = req_Trace_End_Sec_Name

        self.req_trace_tables = None
        self.req_description_tables = None

        self.__getTables()

    def __getTables(self):
        docxParser = DocxParser(self.filePath)
        self.req_trace_tables = docxParser.extract_tables_between_titles(self.req_Trace_Start_Sec_Name,
                                                                         self.req_Trace_End_Sec_Name)
        self.sw_req_tables = docxParser.extract_tables_between_titles(self.req_Start_Sec_Name,
                                                                      self.req_end_Sec_Name)
        self.__generateSRS_outputs()

    # Function to normalize column names by removing all kinds of whitespace
    def __normalize_column_name(self, col_name):
        return re.sub(r'\s+', '', col_name)

    def __generateSRS_outputs(self):
        for sw_req_table in self.sw_req_tables:
            transposed_df = sw_req_table.transpose()
            transposed_df.columns = transposed_df.iloc[0]
            preprocessed_df = transposed_df.drop(0)
            # Normalize column names by removing all kinds of whitespace
            preprocessed_df.columns = [self.__normalize_column_name(col) for col in preprocessed_df.columns]

            # Filter columns based on keywords
            keyword_columns = {keyword: [col for col in preprocessed_df.columns if keyword in col] for keyword in self.keywords}

            # Check if all keywords are present
            if all(keyword_columns[keyword] for keyword in self.keywords):
                id = preprocessed_df.iloc[0][self.keywords[0]]
                description = preprocessed_df.iloc[0][self.keywords[1]]
                self.hdd_outputs[id] = Output(id, description, self.outputType)


    def getReqTrace(self):
        req_trace_df = pd.DataFrame()
        for req_trace_table in self.req_trace_tables:
            req_trace_table.columns = req_trace_table.iloc[0]
            preprocessed_req_trace_df = req_trace_table.drop(0)
            if req_trace_df.empty:
                req_trace_df = preprocessed_req_trace_df
            else:
                req_trace_df = pd.concat([req_trace_df, preprocessed_req_trace_df], ignore_index=True)
        return req_trace_df

    def getSRS_outputs(self):
        return self.hdd_outputs
