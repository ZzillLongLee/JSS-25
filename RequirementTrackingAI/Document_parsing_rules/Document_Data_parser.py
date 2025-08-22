from Output_generation.OutputGenerator import OutputGenerator
from Document_parsing_rules.SRS import SRS_Document
from Document_parsing_rules.SSS import SSS_Document
from Document_parsing_rules.SSDD import SSDD_Document
from Document_parsing_rules.HRS import HRS_Document
from Document_parsing_rules.HDD import HDD_Document
import pandas as pd
import os

Requirements = []
ssdd_outputs = []
hrs_outputs = []
hdd_outputs = []

def search_and_collect_outputs(directory, search_characters, type):
    req_trace_df = None
    total_descriptions = []
    # Search for files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if search_characters in file:
                file_path = os.path.join(root, file)
                print(f"Opening file: {file_path}")
                try:
                    if type == 1:
                        hrs = HRS_Document(file_path, '하드웨어 형상항목 요구사항', '요구사항의 우선순위',
                                           '요구사항 추적성', '참고사항')
                        temp_req_trace_df = hrs.getReqTrace()
                        temp_descriptions = hrs.getReqDescriptions()
                        total_descriptions.extend(temp_descriptions)
                        # Concatenate the dataframe vertically
                        if req_trace_df is None:
                            req_trace_df = temp_req_trace_df
                        else:
                            req_trace_df = pd.concat([req_trace_df, temp_req_trace_df], ignore_index=True)

                    if type == 2:
                        hdd = HDD_Document(file_path, '형상품목 상세설계', '요구사항 추적성', '요구사항 추적성',
                                           '참고사항')
                        temp_req_trace_df = hdd.getReqTrace()
                        temp_outputs = hdd.getHDD_outputs()
                        total_descriptions.append(temp_outputs)
                        # Concatenate the dataframe vertically
                        if req_trace_df is None:
                            req_trace_df = temp_req_trace_df
                        else:
                            req_trace_df = pd.concat([req_trace_df, temp_req_trace_df], ignore_index=True)

                except Exception as e:
                    print(f"Failed to open {file_path}: {e}")

    return req_trace_df, total_descriptions


if __name__ == '__main__':

    outputGen = OutputGenerator()
    document_folder_path = r'C:\Users\DTaQ\Desktop\My task\2024 personal project\2024 연구그룹 과제\SRS'
    SRS_file_name = r'\소프트웨어요구사항명세서_신호발생기.docx'

    srs = SRS_Document(document_folder_path + SRS_file_name, '소프트웨어 형상항목 요구사항', '요구사항의 우선순위', '요구사항 추적성', '참고사항', 'SRS')
    srs_outputs = srs.getSRS_outputs()
    srs_reqTrace = srs.getReqTrace()
    print()

