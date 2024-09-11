# Cross Validation Classification LogLoss
import pandas as pd
from tqdm import tqdm
import os
import queryLlama3 as ql
import pdb
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

def combinations(row):
    premise1 = ' '.join([str(row['Antecedent']), str(row['Antecedent_kg']), str(row['Consequent_kg'])])
    premise2 = ' '.join([str(row['Antecedent']), str(row['Antecedent_kg'])])
    premise3 = ' '.join([str(row['Antecedent']), str(row['Consequent_kg'])])
    conclusion1 = ' '.join([row['Consequent'], str(row['Antecedent_kg']), str(row['Consequent_kg'])])
    conclusion2 = ' '.join([row['Consequent'], str(row['Antecedent_kg'])])
    conclusion3 = ' '.join([row['Consequent'], str(row['Consequent_kg'])])

    return premise1, premise2, premise3, conclusion1, conclusion2, conclusion3
def queryCCC_entailment(row):

    premise1, premise2, premise3, conclusion1, conclusion2, conclusion3 = combinations(row)
    template = '''Decide whether Claim2 follows Claim1, i.e. Claim1 entails or infers Claim2
Claim1: {}
Claim2: {}

Answer in JSON format with fields "Answer" and "Explanation". In "Answer", use 1 for Yes or -1 for No or 0 for unsure.'''

    output = [  ql.query(template.format(row['Antecedent'], row['Consequent'])),
                ql.query(template.format(premise1, row['Consequent'])),
                ql.query(template.format(premise1, conclusion1)),
                ql.query(template.format(premise1, conclusion2)),
                ql.query(template.format(premise1, conclusion3)),  # current best
                ql.query(template.format(premise2, row['Consequent'])),
                ql.query(template.format(premise2, conclusion1)),
                ql.query(template.format(premise2, conclusion2)),
                ql.query(template.format(premise2, conclusion3)),
                ql.query(template.format(premise3, row['Consequent'])),
                ql.query(template.format(premise3, conclusion1)),
                ql.query(template.format(premise3, conclusion2)),
                ql.query(template.format(premise3, conclusion3))]

    return output

def queryCCC_verifiable(row):
    premise1, premise2, premise3, conclusion1, conclusion2, conclusion3 = combinations(row)
    template = '''A claim is verifiable if its truth value can be derived or tested to be true or false based on specified knowledge. Is Claim1 verifiable? Claim1 is originally from the CC in an online conversion. Claim1: {} 
    CC: {}
    Answer in JSON format with fields "Answer" and "Explaination". In "Answer", use 1 for Yes or -1 for No or 0 for unsure. '''

    output = [  ql.query(template.format(row['Antecedent'], row['Consequent']))["Answer"],
                ql.query(template.format(premise1, row['Consequent']))["Answer"],
                ql.query(template.format(premise1, conclusion1))["Answer"],
                ql.query(template.format(premise1, conclusion2))["Answer"],
                ql.query(template.format(premise1, conclusion3))["Answer"],  # current best
                ql.query(template.format(premise2, row['Consequent']))["Answer"],
                ql.query(template.format(premise2, conclusion1))["Answer"],
                ql.query(template.format(premise2, conclusion2))["Answer"],
                ql.query(template.format(premise2, conclusion3))["Answer"],
                ql.query(template.format(premise3, row['Consequent']))["Answer"],
                ql.query(template.format(premise3, conclusion1))["Answer"],
                ql.query(template.format(premise3, conclusion2))["Answer"],
                ql.query(template.format(premise3, conclusion3))["Answer"]]

    return output


def queryCCC_sub_claim_truth_value(row):
    premise1, premise2, premise3, conclusion1, conclusion2, conclusion3 = combinations(row)
    template = '''A claim is verifiable if its truth value can be derived or tested to be true or false based on specified knowledge. Is Claim1 verifiable? Claim1 is originally from the CC in an online conversion. Claim1: {} 
    CC: {}
    Answer in JSON format with fields "Answer" and "Explaination". In "Answer", use 1 for Yes or -1 for No or 0 for unsure. '''

    output = [  ql.query(template.format(row['Antecedent'], row['Consequent']))["Answer"],
                ql.query(template.format(premise1, row['Consequent']))["Answer"],
                ql.query(template.format(premise1, conclusion1))["Answer"],
                ql.query(template.format(premise1, conclusion2))["Answer"],
                ql.query(template.format(premise1, conclusion3))["Answer"],  # current best
                ql.query(template.format(premise2, row['Consequent']))["Answer"],
                ql.query(template.format(premise2, conclusion1))["Answer"],
                ql.query(template.format(premise2, conclusion2))["Answer"],
                ql.query(template.format(premise2, conclusion3))["Answer"],
                ql.query(template.format(premise3, row['Consequent']))["Answer"],
                ql.query(template.format(premise3, conclusion1))["Answer"],
                ql.query(template.format(premise3, conclusion2))["Answer"],
                ql.query(template.format(premise3, conclusion3))["Answer"]]

    return output

def run(dataframe, out_file, out_columns, function):
    number_lines = len(dataframe)
    chunksize = 1

    if (out_file is None):
        out_file_valid = False
        already_done = pd.DataFrame().reindex(columns=dataframe.columns)
        start_line = 0

    elif isinstance(out_file, str):
        out_file_valid = True
        if os.path.isfile(out_file):
            already_done = pd.read_csv(out_file)
            start_line = len(already_done)
        else:
            already_done = pd.DataFrame().reindex(columns=dataframe.columns)
            start_line = 0
    else:
        print('ERROR: "out_file" is of the wrong type, expected str')

    for i in tqdm(range(start_line, number_lines, chunksize)):
        sub_df = dataframe.iloc[i: i + chunksize]
        sub_df[out_columns] = sub_df.apply(lambda x: function(x), axis=1, result_type='expand')
        # pdb.set_trace()
        already_done = pd.concat([already_done, sub_df], axis=0)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_csv(out_file)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_pickle(out_file.replace(".csv", '.pkl'))

    return already_done


if __name__ == '__main__':
    input_file = 'data/miscc_tv_both_kg_info.csv'
    # input_File = '/Users/xueli/Library/CloudStorage/OneDrive-UniversityofEdinburgh/code/miscc_1/2024_data/miscc_data/baseline/semval_classification_2.csv'
    out_path = 'data/miscc_tv_both_kg_tv.csv'

    if input_file[-4:] == '.pkl':
        df = pd.read_pickle(input_file)
    elif input_file[-4:] == '.csv':
        df = pd.read_csv(input_file)
    else:
        print("invalid input file")
    record_file = (
        "/Users/xueli/Library/CloudStorage/OneDrive-UniversityofEdinburgh/code/miscc_1/kg_entity_linking/files_record/llama3_entailment.json")
    task_function = queryCCC_entailment
    out_columns = ["env_query_myllama3_" + str(i) for i in range(13)]
    print(out_columns)
    run(df.loc[:, ~df.columns.str.contains('^Unnamed')], out_path, out_columns,
        task_function)
    print('finished')

