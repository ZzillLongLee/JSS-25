from word.DocxParser import DocxParser
from dao.Output import Output
import pandas as pd


class SDD_Document:

    def __init__(self, filePath, design_Start_Sec_Name, design_end_Sec_Name, req_Trace_Start_Sec_Name,
                 req_Trace_End_Sec_Name, outputType):
        self.outputType = outputType
        self.sdd_outputs = {}
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
        self.__generateSDD_Outputs(design_description_tables)
        self.req_trace_tables = req_trace_tables

    def __generateSDD_Outputs(self, design_description_tables):
        for table in design_description_tables:
            table.columns = table.iloc[0]
            preprocessed_df = table.drop(0)
            for idx, row in preprocessed_df.iterrows():
                id = row[1]
                description = row[2]
                self.sdd_outputs[id] = Output(id, description, self.outputType)

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

    def getSDD_outputs(self):
        return self.sdd_outputs
