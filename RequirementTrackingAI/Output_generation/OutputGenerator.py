from Dao.Output import Output


class OutputGenerator:

    def __init__(self):
        self.sss_id_colName = '체계/부체계 규격서 식별자'
        self.ssdd_id_colName = '체계/부체계 설계기술서 식별자'
        self.hrs_id_colName = 'HW 요구사항명세서 식별자'
        self.hdd_id_colName = 'HW 설계기술서 식별자'
        self.hrs_paired_colName = '체계 요구사항 명세서 식별자'
        self.srs_id_colName = 'SW 요구사항 명세서 식별자'
        self.sdd_id_colName = 'SW 설계기술서 식별자'
        self.sdd_paired_colName = 'SW 요구사항명세서 식별자'

    def generate_SSS_Outputs(self, output_trace_df, output_descriptions, outputs, outputType):
        output_ids = output_trace_df['체계/부체계 규격서 식별자']
        output_ids = output_ids.drop_duplicates()
        for output_id in output_ids:
            for output_description in output_descriptions:
                if output_id == str(output_description.loc[1, '식별자']):
                    outputs[output_id] = Output(output_id, str(output_description.loc[1, '요구사항']), outputType)
                    break

    def generate_SSDD_Outputs(self, id, ssdd_design_descriptions, sss_id, outputs, outputType):
        found = False
        for ssdd_design_description in ssdd_design_descriptions:
            for idx, row in ssdd_design_description.iterrows():
                if id == row['설계식별자']:
                    description = row['설계명세']
                    output = Output(id, description, outputType)
                    if '\r' in sss_id:
                        sss_ids = sss_id.split('\r')
                        for splited_sss_id in sss_ids:
                            sss_output = outputs.get(splited_sss_id)
                            if sss_output != None:
                                sss_output.add_child_output(output)
                    else:
                        sss_output = outputs.get(sss_id)
                        if sss_output != None:
                            sss_output.add_child_output(output)

                    outputs[id] = output
                    found = True
                    break
            if found == True:
                break

    def generate_HRS_Outputs(self, id, hrs_req_descriptions, ssdd_id, outputs, outputType):
        found = False
        for hrs_req_description in hrs_req_descriptions:
            for idx, row in hrs_req_description.iterrows():
                if id == row[0]:
                    description = row[1]
                    hrs_output = Output(id, description, outputType)
                    if '\r' in ssdd_id:
                        ssdd_ids = ssdd_id.split('\r')
                        for splited_ssdd_id in ssdd_ids:
                            ssdd_output = outputs.get(splited_ssdd_id)
                            if ssdd_output != None:
                                ssdd_output.add_child_output(hrs_output)
                    else:
                        ssdd_output = outputs.get(ssdd_id)
                        if ssdd_output != None:
                            ssdd_output.add_child_output(hrs_output)

                    outputs[id] = hrs_output
                    found = True
                    break

            if found == True:
                break

    def generate_HDD_Outputs(self, id, hrs_id, outputs):
        hdd_output = outputs.get(id)
        if '\r' in hrs_id:
            hrs_ids = hrs_id.split('\r')
            for hrs_id in hrs_ids:
                hrs_output = outputs.get(hrs_id)
                if hrs_output != None:
                    hrs_output.add_child_output(hdd_output)
        else:
            hrs_output = outputs.get(hrs_id)
            if hrs_output != None:
                hrs_output.add_child_output(hdd_output)

    def generate_SRS_Outputs(self, id, ssdd_id, outputs):
        srs_output = outputs.get(id)
        if '\r' in ssdd_id:
            ssdd_ids = ssdd_id.split('\r')
            for ssdd_id in ssdd_ids:
                ssdd_output = outputs.get(ssdd_id)
                if ssdd_output != None:
                    ssdd_output.add_child_output(srs_output)
        else:
            ssdd_output = outputs.get(ssdd_id)
            if ssdd_output != None:
                ssdd_output.add_child_output(srs_output)

    def generate_SDD_Outputs(self, id, srs_id, outputs):
        sdd_output = outputs.get(id)
        if '\r' in srs_id:
            srs_ids = srs_id.split('\r')
            for srs_id in srs_ids:
                srs_output = outputs.get(srs_id)
                if srs_output != None:
                    srs_output.add_child_output(sdd_output)
        else:
            srs_output = outputs.get(srs_id)
            if srs_output != None:
                srs_output.add_child_output(sdd_output)
