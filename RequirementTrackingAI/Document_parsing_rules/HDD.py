from word.DocxParser import DocxParser
from dao.Output import Output
import pandas as pd


class HDD_Document:

    def __init__(self, filePath, design_Start_Sec_Name, design_End_Sec_Name, design_Trace_Start_Sec_Name,
                 design_Trace_End_Sec_Name, outputType):
        self.outputType = outputType
        self.filePath = filePath
        self.design_Start_Sec_Name = design_Start_Sec_Name
        self.design_End_Sec_Name = design_End_Sec_Name
        self.design_Trace_Start_Sec_Name = design_Trace_Start_Sec_Name
        self.design_Trace_End_Sec_Name = design_Trace_End_Sec_Name

        self.design_trace_tables = None
        self.design_description_tables = None

        self.docx_parser = DocxParser(self.filePath)
        self.req_trace_tables = self.docx_parser.extract_tables_between_titles(self.design_Trace_Start_Sec_Name,
                                                                               self.design_Trace_End_Sec_Name)
        self.req_trace_table = self.__getReqTrace()
        self.hdd_outputs = self.generateHDD_outputs()

    def generateHDD_outputs(self):
        id_list = self.req_trace_table['HW 설계기술서 식별자'].tolist()
        doc = self.docx_parser.getDoc()
        hdd_outputs = {}
        current_id = None
        collecting = False
        description = []

        table_markers = ["\r\x07", "\x07"]
        concatenated_text = ""
        collecting_table = False

        for i in range(1, doc.Paragraphs.Count + 1):
            paragraph = doc.Paragraphs(i)
            text = paragraph.Range.Text.strip()

            if self.design_Start_Sec_Name in text:
                collecting = True
                description = []
            elif collecting and self.design_End_Sec_Name in text:
                if current_id is not None:
                    description_str = self.generateDescription(description)
                    hdd_outputs[current_id] = Output(current_id, description_str, self.outputType)
                collecting = False
            elif collecting and text:
                if any(doc_id in text for doc_id in id_list):
                    if current_id is not None:
                        description_str = self.generateDescription(description)
                        hdd_outputs[current_id] = Output(current_id, description_str, self.outputType)
                    current_id = next((doc_id for doc_id in id_list if doc_id in text), None)
                    description = []
                elif any(marker in text for marker in table_markers):
                    cleaned_text = self.__clean_text(text)
                    concatenated_text += cleaned_text
                    collecting_table = True
                else:
                    if collecting_table:
                        concatenated_text = concatenated_text.strip()
                        self.__compareTable(concatenated_text, doc, description)
                        concatenated_text = ""
                        collecting_table = False
                    description.append(('paragraph', text))

        if collecting and current_id is not None:
            description_str = self.generateDescription(description)
            hdd_outputs[current_id] = Output(current_id, description_str, self.outputType)

        return hdd_outputs

    def __compareTable(self, concatenated_text, doc, description):
        for table in doc.Tables:
            table_str = self.__table_to_str(table)
            if concatenated_text == table_str:
                description.append(('table', table))
                break

    def __getReqTrace(self):
        req_trace_df = pd.DataFrame()
        for req_trace_table in self.req_trace_tables:
            req_trace_table.columns = req_trace_table.iloc[0]
            preprocessed_req_trace_df = req_trace_table.drop(0)
            if req_trace_df.empty:
                req_trace_df = preprocessed_req_trace_df
            else:
                req_trace_df = pd.concat([req_trace_df, preprocessed_req_trace_df], ignore_index=True)
        return req_trace_df

    def getReqTrace(self):
        return self.req_trace_table

    def getHDD_outputs(self):
        return self.hdd_outputs

    def __table_to_str(self, table):
        table_str = ''
        for row in range(1, table.Rows.Count + 1):
            for col in range(1, table.Columns.Count + 1):
                try:
                    cell = table.Cell(row, col)
                    cell_text = cell.Range.Text.strip().replace('\r\x07', '').replace('\x07', '')
                    table_str += cell_text
                except Exception as e:
                    table_str += ''
        return table_str.strip()

    def __clean_text(self, text):
        return text.replace('\r\x07', '').replace('\x07', '').strip()

    def generateDescription(self, description):
        result_str = ""
        for item in description:
            item_type, value = item
            if item_type == 'paragraph':
                result_str += value + "\n\n"
            elif item_type == 'table':
                result_str += self.__table_to_str(value) + "\n"

        return result_str