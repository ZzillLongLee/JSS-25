from itertools import product
import pandas as pd

class DatasetGenerator:

    def expand_ids(self, df, column):
        df.columns = df.columns.str.replace('\r', ' ')
        expanded_rows = []
        for index, row in df.iterrows():
            idValues = row[column]
            ids = None
            if '\n' in idValues:
                ids = row[column].split('\n')
            if '\r' in idValues:
                ids = row[column].split('\r')
            if ids != None:
                for id_ in ids:
                    expanded_row = row.copy()
                    expanded_row[column] = id_
                    expanded_rows.append(expanded_row)
            else:
                row = row.copy()
                expanded_rows.append(row)

        return pd.DataFrame(expanded_rows)

    # Function to split and reformat the DataFrame
    def split_rows(self, df, outputs, columnName_1, columnName_2):
        new_rows = []
        for index, row in df.iterrows():
            col1_splits = str(row[columnName_1]).splitlines()
            col2_splits = str(row[columnName_2]).splitlines()

            max_splits = max(len(col1_splits), len(col2_splits))
            for i in range(max_splits):
                new_row = [
                    col1_splits[i] if i < len(col1_splits) else '',
                    col2_splits[i] if i < len(col2_splits) else ''
                ]
                new_rows.append(new_row)

        # Create a new DataFrame from the new rows
        new_df = pd.DataFrame(new_rows, columns=df.columns)
        return new_df

    def pair_all_with_all_and_label2(self, df, outputs, columnName_1, columnName_2):
        # Expand the IDs in both columns
        step1_expanded = self.expand_ids(df, columnName_1)
        step2_expanded = self.expand_ids(df, columnName_2)

        # Create the cartesian product for all pairs
        cartesian_product = pd.DataFrame(list(product(step1_expanded[columnName_1], step2_expanded[columnName_2])),
                                         columns=[columnName_1, columnName_2])

        new_dict = {id: value for id, value in outputs.items()}

        # Add the Step1_Output and Step2_Output based on the dictionaries
        cartesian_product['src_sentence'] = cartesian_product[columnName_1].map(new_dict)
        cartesian_product['tar_sentence'] = cartesian_product[columnName_2].map(new_dict)

        # Add the label column
        original_pairs = set(
            (sid, tid)
            for sids, tids in zip(df[columnName_1], df[columnName_2])
            for sid in sids.split('\r')
            for tid in tids.split('\r')
        )
        cartesian_product['Label'] = cartesian_product.apply(
            lambda row: (row[columnName_1], row[columnName_2]) in original_pairs, axis=1
        )
        clean_cartesian_product = cartesian_product.dropna()
        # Select only the required columns
        result = clean_cartesian_product[['src_sentence', 'tar_sentence', 'Label']]

        return result


    def pair_all_with_all_and_label(self, df, outputs, columnName_1, columnName_2):
        # Expand the IDs in both columns
        step1_expanded = self.expand_ids(df, columnName_1)
        step2_expanded = self.expand_ids(df, columnName_2)

        # Create the cartesian product for all pairs
        cartesian_product = pd.DataFrame(list(product(step1_expanded[columnName_1], step2_expanded[columnName_2])),
                                         columns=[columnName_1, columnName_2])

        new_dict = {id: details.getDescription() for id, details in outputs.items()}

        # Add the Step1_Output and Step2_Output based on the dictionaries
        cartesian_product['src_sentence'] = cartesian_product[columnName_1].map(new_dict)
        cartesian_product['tar_sentence'] = cartesian_product[columnName_2].map(new_dict)

        # Add the label column
        original_pairs = set(
            (sid, tid)
            for sids, tids in zip(df[columnName_1], df[columnName_2])
            for sid in sids.split('\r')
            for tid in tids.split('\r')
        )
        cartesian_product['Label'] = cartesian_product.apply(
            lambda row: (row[columnName_1], row[columnName_2]) in original_pairs, axis=1
        )
        clean_cartesian_product = cartesian_product.dropna()
        # Select only the required columns
        result = clean_cartesian_product[['src_sentence', 'tar_sentence', 'Label']]

        return result


if __name__ == '__main__':
    data = {
        'Step1_ID': ['A1\nA2', 'B3', 'C4\nC5'],
        'Step2_ID': ['D1', 'E2\nE3', 'F4']
    }

    step1_outputs = {'A1': 'OutputA1', 'A2': 'OutputA2', 'B3': 'OutputB3', 'C4': 'OutputC4', 'C5': 'OutputC5'}
    step2_outputs = {'D1': 'OutputD1', 'E2': 'OutputE2', 'E3': 'OutputE3', 'F4': 'OutputF4'}
    step1_outputs.update(step2_outputs)

    df = pd.DataFrame(data)

    dg = DatasetGenerator()
    # Apply the function to pair all Step1_IDs with all Step2_IDs and add the label
    result_df = dg.pair_all_with_all_and_label(df, step1_outputs, 'Step1_ID', 'Step2_ID')
    print(result_df)
