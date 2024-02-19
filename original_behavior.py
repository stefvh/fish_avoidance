def format_io_poses(df_poses, time_points, type="robot"):
    ids = np.unique(df_poses['id'])
    print(f"IDs: {ids}, type: {type}")
    if type == "robot":
        k_dim = 4
    elif type == "fish":
        k_dim = 4
    else:
        raise ValueError("Invalid type")
    io_poses = np.zeros((len(ids), len(time_points), k_dim))
    nan_time_points = {}
    for j, identifier in enumerate(ids):
        nan_time_points[identifier] = []
        for i, time in enumerate(time_points):
            df = df_poses.loc[
                (df_poses['time'] == time) & 
                (df_poses['id'] == identifier)
            ]
            if df.empty:
                io_poses[j, i] = np.float32([np.nan] * k_dim)
                nan_time_points[identifier].append(i)
            else:
                io_poses[j, i, 0] = max(min(float(df['x'].iloc[0] - 50.), 50.45), -50.45)
                io_poses[j, i, 1] = max(min(float(df['y'].iloc[0] - 50.), 50.45), -50.45)
                if k_dim == 4:
                    io_poses[j, i, 2] = np.cos((df['z'].iloc[0]))
                    io_poses[j, i, 3] = np.sin((df['z'].iloc[0]))
                elif type == 3:
                    io_poses[j, i, 2] = float(df['z'].iloc[0]) % (2 * np.pi)
                else:
                    raise ValueError("Invalid type")
        # if type == "fish":
        #     print(f"Fish {identifier}:")
        #     df = df_poses.loc[
        #         (df_poses['id'] == identifier)
        #     ]
        #     print(df.shape)
        #     # print first 20 rows
        #     # print(df.head(20))
        #     # print(df[df.isna().any(axis=1)])
        #     nans = np.argwhere(np.isnan(io_poses[j, :, 0]))
        #     # print(nans)
        #     print(nans.shape)

    return io_poses