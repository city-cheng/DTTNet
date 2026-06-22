import numpy as np


def sample_with_jump(num_frames, length, max_jump, mode="frame"):
    frames_idx = None
    if mode == "frame":
        this_max_jump = min(length, max_jump)
        # iterative sampling
        frames_idx = [np.random.randint(length - num_frames + 1)]
        # print(f'frames_idx:{frames_idx}')
        acceptable_set = set(
            range(
                max(0, frames_idx[-1] - this_max_jump),
                min(length, frames_idx[-1] + this_max_jump + 1),
            )
        ).difference(set(frames_idx))
        # print(f'acceptable_set:{acceptable_set}')
        while len(frames_idx) < num_frames:
            idx = np.random.choice(list(acceptable_set))
            # print(f'idx:{idx}')
            frames_idx.append(idx)
            new_set = set(
                range(
                    max(0, frames_idx[-1] - this_max_jump),
                    min(length, frames_idx[-1] + this_max_jump + 1),
                )
            )
            # print(f'new_set:{new_set}')
            acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))
            # print(f'new acceptable_set:{acceptable_set}')
            # print('*'*10)
    elif mode == "clip":
        frames_idx = [np.random.randint(0, length - num_frames + 1)]
        stride = np.random.randint(num_frames, max_jump + 1)
        if frames_idx[-1] + stride <= length - num_frames:
            frames_idx.append(frames_idx[-1] + stride)
        else:
            frames_idx.append(frames_idx[-1] - num_frames)

        return [
            list(range(frames_idx[0], frames_idx[0] + 5)),
            list(range(frames_idx[1], frames_idx[1] + 5)),
        ]

    frames_idx = sorted(frames_idx)
    if np.random.rand() < 0.5:
        # 一半的几率反转片段
        frames_idx = frames_idx[::-1]

    return frames_idx
