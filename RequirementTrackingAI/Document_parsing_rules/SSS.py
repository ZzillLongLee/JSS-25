from word.DocxParser import DocxParser


class SSS_Document:

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
        req_trace_df = self.req_trace_tables[0]
        req_trace_df = req_trace_df.drop(req_trace_df.index[[0,1]])
        req_trace_df.columns = ['출처', '내용', '체계/부체계 규격서 식별자']
        return req_trace_df

    def getReqDescriptions(self):
        return self.req_description_tables
