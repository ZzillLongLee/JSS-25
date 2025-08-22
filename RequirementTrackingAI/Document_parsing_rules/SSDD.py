from word.DocxParser import DocxParser
import pandas as pd

class SSDD_Document:

    def __init__(self, filePath, design_Start_Sec_Name, design_end_Sec_Name, req_Trace_Start_Sec_Name,
                 req_Trace_End_Sec_Name):
        self.filePath = filePath
        self.design_Start_Sec_Name = design_Start_Sec_Name
        self.design_end_Sec_Name = design_end_Sec_Name
        self.req_Trace_Start_Sec_Name = req_Trace_Start_Sec_Name
        self.req_Trace_End_Sec_Name = req_Trace_End_Sec_Name

        self.req_trace_tables = None
        self.design_description_tables = None

        self.__getTables()

    def __getTables(self):
        docxParser = DocxParser(self.filePath)
        req_trace_tables = docxParser.extract_tables_between_titles(self.req_Trace_Start_Sec_Name,
                                                                    self.req_Trace_End_Sec_Name)
        design_description_tables = docxParser.extract_tables_between_titles(self.design_Start_Sec_Name,
                                                                          self.design_end_Sec_Name)
        preprocessed_design_description_tables = self.__preprocess_Design_Description_tables(design_description_tables)
        self.req_trace_tables = req_trace_tables
        self.design_description_tables = preprocessed_design_description_tables

    def __preprocess_Design_Description_tables(self, design_description_tables):
        preprocessed_tables = []
        for table in design_description_tables:
            value_to_find = ['설계식별자', '설계명세']
            preprocessed_df = self.find_set_Column(table, value_to_find)
            # Check if any value in the DataFrame contains '\r'
            if any('\r' in str(cell) for row in preprocessed_df.values for cell in row):
                preprocessed_df = preprocessed_df.replace('\r', '', regex=True)
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

    def getDesignDescriptions(self):
        return self.design_description_tables

    def find_set_Column(self, table, value_to_find):
        tar_index = None
        temp_table = table
        for index, row in temp_table.iterrows():
            if value_to_find[0] in row.values and value_to_find[1] in row.values:
                tar_index = index
        if tar_index != None:
            table.columns = table.iloc[tar_index]
            table = table.iloc[tar_index+1:, :]
        return table