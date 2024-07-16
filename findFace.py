import wx
import cv2
import numpy as np
import time
import dlib
import threading


class FaceRecognitionApp(wx.Frame):
    def __init__(self, parent, title):
        super(FaceRecognitionApp, self).__init__(parent, title=title, size=(800, 600))

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.load_image_btn = wx.Button(panel, label='Load Reference Image')
        self.load_video_btn = wx.Button(panel, label='Load Video')
        self.play_pause_btn = wx.Button(panel, label='Play')
        self.forward_btn = wx.Button(panel, label='Forward')
        self.backward_btn = wx.Button(panel, label='Backward')
        self.video_panel = wx.Panel(panel)
        self.video_panel.SetBackgroundColour(wx.BLACK)
        self.video_bitmap = wx.StaticBitmap(self.video_panel)
        self.slider = wx.Slider(panel, style=wx.SL_HORIZONTAL)

        hbox_controls = wx.BoxSizer(wx.HORIZONTAL)
        hbox_controls.Add(self.play_pause_btn, flag=wx.RIGHT, border=5)
        hbox_controls.Add(self.forward_btn, flag=wx.RIGHT, border=5)
        hbox_controls.Add(self.backward_btn, flag=wx.RIGHT, border=5)

        vbox.Add(self.video_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.slider, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(hbox_controls, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.load_image_btn, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.load_video_btn, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(vbox)

        self.Bind(wx.EVT_BUTTON, self.load_reference_image, self.load_image_btn)
        self.Bind(wx.EVT_BUTTON, self.load_video, self.load_video_btn)
        self.Bind(wx.EVT_BUTTON, self.on_play_pause, self.play_pause_btn)
        self.Bind(wx.EVT_BUTTON, self.on_forward, self.forward_btn)
        self.Bind(wx.EVT_BUTTON, self.on_backward, self.backward_btn)
        self.Bind(wx.EVT_SLIDER, self.on_seek, self.slider)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_update)

        self.reference_image_path = None
        self.video_path = None
        self.reference_face_descriptor = None

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("data/dlib/shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("data/dlib/dlib_face_recognition_resnet_model_v1.dat")

        self.SetSizeHints(800, 600, 800, 600)
        self.Centre()
        self.Show()

        self.cap = None
        self.playing = False
        self.tracker = dlib.correlation_tracker()
        self.tracking_face = False
        self.match_found = False
        self.match_start_time = 0
        self.frame_count = 0
        self.detection_interval = 5  # Detect every 5 frames
        self.dist = 1

    def load_reference_image(self, event):
        openFileDialog = wx.FileDialog(self, "Open Reference Image", "", "",
                                       "Image files (*.jpg;*.png)|*.jpg;*.png",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return

        self.reference_image_path = openFileDialog.GetPath()
        self.load_face_descriptor()

    def load_face_descriptor(self):
        if self.reference_image_path:
            self.reference_face_descriptor, self.reference_face_img = self.compute_face_descriptor(
                self.reference_image_path)
            if self.reference_face_descriptor is None:
                wx.MessageBox("Failed to load external face descriptor.", "Error", wx.OK | wx.ICON_ERROR)
            # else:
            #     wx.MessageBox("Reference image loaded successfully!", "Success", wx.OK | wx.ICON_INFORMATION)

    def compute_face_descriptor(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = self.detector(img_rgb, 1)

        if len(dets) == 0:
            print(f"Expected one face in {image_path}, but found {len(dets)}")
            return None, img

        shape = self.sp(img_rgb, dets[0])
        face_descriptor = self.facerec.compute_face_descriptor(img_rgb, shape)
        return np.array(face_descriptor), img_rgb

    def load_video(self, event):
        openFileDialog = wx.FileDialog(self, "Open Video", "", "",
                                       "Video files (*.mp4;*.avi)|*.mp4;*.avi",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return

        self.video_path = openFileDialog.GetPath()
        wx.MessageBox("Video file loaded successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.slider.SetRange(0, self.total_frames)

    def on_play_pause(self, event):
        if not self.cap:
            wx.MessageBox("Please load the video file first.", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.playing = not self.playing
        self.play_pause_btn.SetLabel("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.Start(1000 // self.frame_rate)
        else:
            self.timer.Stop()

    def on_forward(self, event):
        if self.cap:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(current_frame + self.frame_rate * 10, self.total_frames))
            self.slider.SetValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def on_backward(self, event):
        if self.cap:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(current_frame - self.frame_rate * 10, 0))
            self.slider.SetValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def on_seek(self, event):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.slider.GetValue())
            self.playing = False
            self.play_pause_btn.SetLabel("Play")
            self.timer.Stop()

    def on_update(self, event):
        if not self.cap or not self.playing:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.tracking_face or self.frame_count % self.detection_interval == 0:
            try:
                dets = self.detector(frame_rgb, 0)
            except Exception as e:
                print(e)

            for d in dets:
                shape = self.sp(frame_rgb, d)
                face_descriptor = self.facerec.compute_face_descriptor(frame_rgb, shape)
                face_descriptor = np.array(face_descriptor)

                self.dist = np.linalg.norm(self.reference_face_descriptor - face_descriptor)
                if self.dist < 0.4:
                    self.tracker.start_track(frame_rgb, d)
                    self.tracking_face = True
                    self.match_found = True
                    self.match_start_time = time.time()
                    break

        if self.tracking_face:
            self.tracker.update(frame_rgb)
            pos = self.tracker.get_position()
            pt1 = (int(pos.left()), int(pos.top()))
            pt2 = (int(pos.right()), int(pos.bottom()))

            match_percentage = (1 - self.dist) * 100
            cv2.rectangle(frame_rgb, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"Matched: {match_percentage:.2f}%", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

            if time.time() - self.match_start_time > 5:
                self.tracking_face = False
                self.match_found = False

        self.frame_count += 1

        combined_image = self.create_combined_image(frame_rgb)

        wx.CallAfter(self.update_video_frame, combined_image)

        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.SetValue(current_frame)
        wx.CallAfter(self.update_time_display, current_frame)

    def create_combined_image(self, frame):
        external_face_img = cv2.resize(self.reference_face_img, (128, 128))
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        new_w = 640
        new_h = int(new_w / aspect_ratio)

        if new_h > 480:
            new_h = 480
            new_w = int(new_h * aspect_ratio)

        resized_frame = cv2.resize(frame, (new_w, new_h))
        combined_image = np.zeros((480, 192 + new_w, 3), dtype=np.uint8)
        combined_image[20:148, 20:148] = external_face_img
        combined_image[:new_h, 192:192 + new_w] = resized_frame
        # cv2.putText(combined_image, "Press 'p' to pause/resume", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (255, 255, 255), 1)
        return combined_image

    def update_video_frame(self, combined_image):
        height, width = combined_image.shape[:2]
        bitmap = wx.Bitmap.FromBuffer(width, height, combined_image)
        self.video_bitmap.SetBitmap(bitmap)
        self.video_panel.Layout()

    def update_time_display(self, current_frame):
        total_seconds = current_frame // self.frame_rate
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        self.slider.SetToolTip(f"{minutes:02}:{seconds:02}")


if __name__ == '__main__':
    app = wx.App()
    FaceRecognitionApp(None, title='Face Recognition')
    app.MainLoop()
