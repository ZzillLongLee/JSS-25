from Word.DocxParser import DocxParser
import pandas as pd

class HRS_Document:

    def __init__(self, filePath, req_Start_Sec_Name, req_end_Sec_Name, req_Trace_Start_Sec_Name,
                 req_Trace_End_Sec_Name):
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
        req_trace_tables = docxParser.extract_tables_between_titles(self.req_Trace_Start_Sec_Name,
                                                                    self.req_Trace_End_Sec_Name)
        req_description_tables = docxParser.extract_tables_between_titles(self.req_Start_Sec_Name,
                                                                          self.req_end_Sec_Name)
        req_description_tables = self.__preprocess_Req_Description_tables(req_description_tables)
        self.req_trace_tables = req_trace_tables
        self.req_description_tables = req_description_tables

    def __preprocess_Req_Description_tables(self, req_description_tables):
        preprocessed_tables = []
        for table in req_description_tables:
            transposed_df = table.transpose()
            # Rename columns using the values in the first row
            transposed_df.columns = transposed_df.iloc[0]
            preprocessed_df = transposed_df.drop(0)
            preprocessed_tables.append(preprocessed_df)
        return preprocessed_tables

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


    def getReqDescriptions(self):
        return self.req_description_tables