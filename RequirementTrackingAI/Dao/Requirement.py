from dao.Output import Output
# Define Requirement class
class Requirement(Output):
    def __init__(self, output_id, description):
        super().__init__(output_id, description, output_type='SSS')