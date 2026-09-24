import cv2


def open_video(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    return video


def get_video_info(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }
