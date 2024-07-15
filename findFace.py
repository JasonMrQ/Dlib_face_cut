import dlib
import cv2
import numpy as np
import time


def load_face_descriptor(image_path, detector, sp, facerec):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_rgb, 1)

    if len(dets) == 0:
        print(f"Expected one face in {image_path}, but found {len(dets)}")
        return None

    shape = sp(img_rgb, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)
    return np.array(face_descriptor)


def main():
    # Path to the external face image
    external_face_path = "data/images/faces_for_test/rain1.jpg"

    # Load external face image and get its descriptor
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("data/dlib/shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("data/dlib/dlib_face_recognition_resnet_model_v1.dat")

    external_face_descriptor = load_face_descriptor(external_face_path, detector, sp, facerec)

    if external_face_descriptor is None:
        print("Failed to load external face descriptor.")
        return

    # Open video stream
    video_src = 'data/videos/rain.mp4'
    cap = cv2.VideoCapture(video_src)  # Change to 0 for webcam

    tracker = dlib.correlation_tracker()
    tracking_face = False
    match_found = False
    match_start_time = 0
    frame_count = 0
    detection_interval = 5  # 每隔5帧进行一次检测

    # 调整处理速度倍数
    speed_up_factor = 8  # 2倍速播放
    frame_skip = speed_up_factor - 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if video_src == 0:
            frame = cv2.flip(frame, 1)

        if not tracking_face or frame_count % detection_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                dets = detector(frame_rgb, 0)  # 设置检测器参数，减少检测时间
            except Exception as e:
                print(e)

            for d in dets:
                shape = sp(frame_rgb, d)
                face_descriptor = facerec.compute_face_descriptor(frame_rgb, shape)
                face_descriptor = np.array(face_descriptor)

                # Compute the Euclidean distance between the external face and the current frame face
                dist = np.linalg.norm(external_face_descriptor - face_descriptor)

                # print('dist=', dist)
                if dist < 0.4:  # Threshold for face matching
                    tracker.start_track(frame_rgb, d)
                    tracking_face = True
                    match_found = True
                    match_start_time = time.time()
                    break

        if tracking_face:
            tracker.update(frame_rgb)
            pos = tracker.get_position()
            pt1 = (int(pos.left()), int(pos.top()))
            pt2 = (int(pos.right()), int(pos.bottom()))

            match_percentage = (1 - dist) * 100
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, f"Matched: {match_percentage:.2f}%", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # 如果匹配时间超过5秒，重新进行检测
            if time.time() - match_start_time > 5:
                tracking_face = False
                match_found = False

        frame_count += 1

        # 跳过帧以实现加速
        for _ in range(frame_skip):
            cap.grab()

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
