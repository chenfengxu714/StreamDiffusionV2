from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os


def store_arrays_to_lmdb(env, arrays_dict, start_index=0):
    """
    Store rows of multiple numpy arrays in a single LMDB.
    Each row is stored separately with a naming convention.
    """
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            for i, row in enumerate(array):
                # Convert row to bytes
                if isinstance(row, str):
                    row_bytes = row.encode()
                else:
                    row_bytes = row.tobytes()
                data_key = f'{array_name}_{start_index + i}_data'.encode()
                txn.put(data_key, row_bytes)


def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))
    return image_shape


def process_data_dict(data_dict, seen_prompts):
    output_dict = {}

    all_videos = []
    all_prompts = []
    for prompt, video in data_dict.items():
        if prompt in seen_prompts:
            continue
        else:
            seen_prompts.add(prompt)

        video = video.half().numpy()
        all_videos.append(video)
        all_prompts.append(prompt)

    if len(all_videos) == 0:
        return {"latents": np.array([]), "prompts": np.array([])}

    all_videos = np.concatenate(all_videos, axis=0)
    output_dict['latents'] = all_videos
    output_dict['prompts'] = np.array(all_prompts)

    return output_dict


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if dtype == str:
        array = row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)

    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array


def main():
    """
    Aggregate all ode pairs inside a folder into a lmdb dataset.
    Each pt file should contain a (key, value) pair representing a
    video's ODE trajectories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        required=True, help="path to ode pairs")
    parser.add_argument("--lmdb_path", type=str,
                        required=True, help="path to lmdb")

    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_path, "*.pt")))

    # figure out the maximum map size needed
    total_array_size = 5000000000000  # adapt to your need, set to 5TB by default

    env = lmdb.open(args.lmdb_path, map_size=total_array_size * 2)

    counter = 0

    seen_prompts = set()  # for deduplication

    for index, file in tqdm(enumerate(all_files)):
        # read from disk
        data_dict = torch.load(file)

        data_dict = process_data_dict(data_dict, seen_prompts)

        # write to lmdb file
        store_arrays_to_lmdb(env, data_dict, start_index=counter)
        counter += len(data_dict['prompts'])

    # save each entry's shape to lmdb
    with env.begin(write=True) as txn:
        for key, val in data_dict.items():
            print(key, val)
            array_shape = np.array(val.shape)
            array_shape[0] = counter

            shape_key = f"{key}_shape".encode()
            shape_str = " ".join(map(str, array_shape))
            txn.put(shape_key, shape_str.encode())


if __name__ == "__main__":
    main()
