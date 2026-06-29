import cv2
import numpy as np
import cv2
import imageio

def crop_mp4_to_mp4_h264(input_path, output_path,
                         fps=None,
                         crop_rect=None,
                         zoom_factor=1.0):
    """
    裁剪 MP4，并重新保存为 Windows 更容易打开的 H.264 MP4
    """

    if not output_path.lower().endswith(".mp4"):
        output_path += ".mp4"

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件: {input_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps is not None else original_fps

    writer = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 裁剪
        if crop_rect:
            x, y, w, h = crop_rect

            frame_height, frame_width = frame.shape[:2]

            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)

            if w <= 0 or h <= 0:
                print(f"错误：无效裁剪区域: {(x, y, w, h)}")
                cap.release()
                return

            frame = frame[y:y+h, x:x+w]

        # 放大
        if zoom_factor != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            frame = cv2.resize(frame, (new_width, new_height))

        # 很重要：H.264 最好使用偶数宽高
        height, width = frame.shape[:2]
        if width % 2 != 0:
            frame = frame[:, :-1]
            width -= 1
        if height % 2 != 0:
            frame = frame[:-1, :]
            height -= 1

        # OpenCV 是 BGR，imageio 需要 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if writer is None:
            writer = imageio.get_writer(
                output_path,
                fps=output_fps,
                codec="libx264",
                pixelformat="yuv420p"
            )

        writer.append_data(frame_rgb)
        frame_count += 1

    cap.release()

    if writer is not None:
        writer.close()
        print(f"处理完成: {output_path}")
        print(f"输出帧数: {frame_count}")
    else:
        print("错误：没有写入任何帧")


def select_crop_rect_from_video(input_path, frame_index=None, display_scale=0.2):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 默认取中间帧，而不是第一帧
    if frame_index is None:
        frame_index = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取指定帧")
        return None

    h, w = frame.shape[:2]
    print(f"原始视频尺寸: {w}x{h}")
    print(f"当前取第 {frame_index} 帧")

    # 缩小显示，防止窗口太大
    display_frame = cv2.resize(
        frame,
        (int(w * display_scale), int(h * display_scale))
    )

    rect = cv2.selectROI(
        "Select crop area",
        display_frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    x, y, rw, rh = rect

    # 把缩小后的坐标换算回原视频坐标
    x = int(x / display_scale)
    y = int(y / display_scale)
    rw = int(rw / display_scale)
    rh = int(rh / display_scale)

    crop_rect = (x, y, rw, rh)

    print(f"crop_rect = {crop_rect}")

    return crop_rect

# 使用示例
if __name__ == "__main__":

    crop_mp4_to_mp4_h264(
        "assets\Safety-critical Interaction_ours.mp4",
        "assets\Safety-critical Interaction_ours_crop.mp4",
        fps=None,  # None 表示使用原视频帧率
        crop_rect=(460, 1120, 2780, 1390),
        zoom_factor=1.0
    )
    #crop_rect = select_crop_rect_from_video("assets\Behavioral Stability_ours.mp4")
