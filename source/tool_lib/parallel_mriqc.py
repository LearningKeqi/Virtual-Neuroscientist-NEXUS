from quality_control_lib import run_mriqc_base
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bids-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temp-work-dir", required=True)
    parser.add_argument("--temp-bidsdb-dir", required=True)

    parser.add_argument("--modality", nargs="+", default=[])

    parser.add_argument("--participant-label", nargs="+", required=True)

    args = parser.parse_args()

    return_result = run_mriqc_base(
        bids_dir=args.bids_dir,
        output_dir=args.output_dir,
        temp_work_dir=args.temp_work_dir,
        temp_bidsdb_dir=args.temp_bidsdb_dir,
        modality=args.modality,
        participant_label=args.participant_label,
    )

    print(return_result)

