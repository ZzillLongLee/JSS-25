# Define Output class
class Output:
    def __init__(self, output_id, description, output_type, node_false_type='a'):
        self.output_id = output_id
        self.description = description
        self.output_type = output_type
        self.node_false_type = node_false_type
        self.category = ''
        self.child_outputs = []

    def add_child_output(self, output):
        if output not in self.child_outputs:
            self.child_outputs.append(output)

    def get_direct_children(self):
        return [child.output_id for child in self.child_outputs]

    def setCategory(self, category):
        self.category = category

    def getCategory(self):
        return self.category

    def getDescription(self):
        return self.description

