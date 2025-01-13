import h5py


def write_element_to_hptr_h5_file(h5_file, group_id, data, metadata):
    with h5py.File(h5_file, "a") as f:
        group = f.create_group(group_id)
        for k, v in data.items():
            group.create_dataset(
                k, data=v, compression="gzip", compression_opts=4, shuffle=True
            )
        # Add metadata to the group
        for k, v in metadata.items():
            group.attrs[k] = v
