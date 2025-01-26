""""""
from datasets import load_dataset, Dataset, IterableDataset, Audio

import logging
from tqdm import tqdm, trange
from pathlib import Path


def add_sample(sample: dict, dataset_dict: dict):
    """Add one sample to dataset, intermediate function for create_dataset_dict()"""
    dataset_dict["Id"].append(sample['client_id'])
    dataset_dict["audio"].append(sample['audio'])
    dataset_dict["sentence"].append(sample['sentence'])


def create_dataset_dict(input_dataset: IterableDataset, number_of_samples: int):
    """Create dataset with given number of samples"""
    dataset_dict = {'Id': [], 'audio': [], 'sentence': []}
    data_iterator = iter(input_dataset)

    for _ in trange(number_of_samples):
        sample = next(data_iterator, None)
        if sample is None:
            break
        add_sample(sample, dataset_dict)

    return dataset_dict


def main(language, count, split, output_parquet):
    # Config logging
    logging.basicConfig(level=logging.INFO)
    # Stream from web
    dataset_name = "mozilla-foundation/common_voice_17_0"
    common_voice_dict = load_dataset(dataset_name, language, split='train', streaming=True, trust_remote_code=True)
    logging.info("Iterator prepared")
    # Create dataset
    dataset_dict = create_dataset_dict(common_voice_dict, count)
    dataset = Dataset.from_dict(dataset_dict)
    logging.info(f"Number of examples: {len(dataset)}")
    # Resample
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    logging.info("Resampling finished")
    """
    # This block doesn't work very well
    # Split
    dataset = dataset.train_test_split(train_size=split) 
    logging.info(f"Split finished with train size: {split}")
    """
    # Save
    save_dir = Path(f"{output_parquet}/{language}.parquet")
    dataset.to_parquet(save_dir)
    logging.info(f"Parquet file created at: {save_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('language',
                        type=str,
                        choices=['fr', 'ja', 'zh'],
                        help="Subsection for a specific language")
    parser.add_argument('-count',
                        type=int,
                        default=300,
                        help="Number of examples in dataset")
    parser.add_argument("-split",
                        type=float,
                        default=0.6,
                        help="Split percentage for train, test will take the rest")
    parser.add_argument('-o', '--output_folder',
                        type=Path,
                        default="../data/",
                        help="Output parquet folder")

    args = parser.parse_args()

    if not args.output_folder.is_dir():
        raise ValueError("Output folder is not a folder")
    if 0 < args.split < 1:
        pass
    else:
        raise ValueError("Split must be between 0 and 1")

    main(language=args.language,
         count=args.count,
         split=args.split,
         output_parquet=args.output_folder)
