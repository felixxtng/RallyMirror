def select_target(tracks):
    if not tracks:
        return None

    target = max(
        tracks,
        key=lambda track: track["bbox"][3]
    )

    return target