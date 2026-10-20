'''
Script to run centerlines_fixed on multiple cases back to back
Set inputs for all cases in the CASES list, and set it to run. Will catch any errors and keep going
'''

import centerlines_fixed
from pathlib import Path


#Set the inputs for the cases that you want to generate centerlines for
#Copy and paste the dictionary for as many cases as you have
CASES = [
    {
        "prep_dir" : "",
        "case_name": "",
        "edge_length": 0.4,
        "iters": 10,
        "ratio": 1.01,
        "resample_step_length": 1.5,
        "fen": True
    },
]

def main():
    for case in CASES:
        print(f'Starting {case["case_name"]}...')
        try:
            centerlines_fixed.make_cl(
                prep_dir=Path(case["prep_dir"]),
                case_name=case["case_name"],
                edge_length=case["edge_length"],
                iters=case["iters"],
                r=case["ratio"],
                resample_step_length=case["resample_step_length"],
                fen=case["fen"]
            )
        except Exception as e:
            print("="*50)
            print(f"ERROR ON CASE {case['case_name']}: {e}")
            print("="*50)
        print(f'Finished {case["case_name"]}!')

if __name__ == "__main__":
    main()

