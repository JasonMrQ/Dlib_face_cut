import os
import re

import wx
import cv2
import numpy as np
import time
import dlib


def validate_input(value):
    pattern = re.compile(r'^(rtsp:|http:|https:)')
    if pattern.match(value):
        return True
    else:
        return False


class FaceRecognitionApp(wx.Frame):
    def __init__(self, parent, title):
        super(FaceRecognitionApp, self).__init__(parent, title=title, size=(800, 800))
        self.SetMinSize((800, 800))
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.load_image_btn = wx.Button(panel, label='Load Image')
        self.load_video_btn = wx.Button(panel, label='Load Video')
        self.play_pause_btn = wx.Button(panel, label='Play')
        self.forward_btn = wx.Button(panel, label='Forward')
        self.backward_btn = wx.Button(panel, label='Backward')
        self.speed_btn = wx.Button(panel, label='Speed x1')
        self.usrCameraCheckbox = wx.CheckBox(panel, label='Use Camera')
        self.video_panel = wx.Panel(panel)
        self.video_panel.SetBackgroundColour(wx.BLACK)
        self.video_bitmap = wx.StaticBitmap(self.video_panel)
        self.slider = wx.Slider(panel, style=wx.SL_HORIZONTAL)
        self.time_display = wx.StaticText(panel, label="00:00 / 00:00")
        self.image_path_tcl = wx.StaticText(panel, label="please load image first.")
        self.video_path_tcl = wx.TextCtrl(panel)
        self.video_path_tcl.SetHint(
            "Supports local device cameras and webcams that support Http|Https|RTSP protocol...")
        # 创建一个 wx.Choice 控件
        choices = ['Local Camera', 'Net Camera']
        self.choice_ctrl = wx.Choice(panel, choices=choices)
        self.choice_ctrl.SetSelection(0)  # 默认选择第一个选项

        hbox_controls = wx.BoxSizer(wx.HORIZONTAL)
        hbox_controls.Add(self.play_pause_btn, flag=wx.RIGHT, border=5)
        hbox_controls.Add(self.forward_btn, flag=wx.RIGHT, border=5)
        hbox_controls.Add(self.backward_btn, flag=wx.RIGHT, border=5)
        hbox_controls.Add(self.speed_btn, flag=wx.RIGHT, border=5)
        hbox_controls.Add(self.usrCameraCheckbox, flag=wx.RIGHT, border=5)
        hbox_controls.Add(self.choice_ctrl, flag=wx.RIGHT, border=5)

        vbox.Add(self.video_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.slider, flag=wx.EXPAND | wx.ALL, border=5)
        vbox.Add(self.time_display, flag=wx.ALIGN_CENTER | wx.ALL, border=5)
        vbox.Add(hbox_controls, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.image_path_tcl, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.load_image_btn, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.video_path_tcl, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(self.load_video_btn, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(vbox)

        self.Bind(wx.EVT_BUTTON, self.load_reference_image, self.load_image_btn)
        self.Bind(wx.EVT_BUTTON, self.load_video, self.load_video_btn)
        self.Bind(wx.EVT_BUTTON, self.on_play_pause, self.play_pause_btn)
        self.Bind(wx.EVT_BUTTON, self.on_forward, self.forward_btn)
        self.Bind(wx.EVT_BUTTON, self.on_backward, self.backward_btn)
        self.Bind(wx.EVT_BUTTON, self.on_speed_change, self.speed_btn)
        self.Bind(wx.EVT_CHECKBOX, self.on_use_camera, self.usrCameraCheckbox)
        self.Bind(wx.EVT_CHOICE, self.on_chioce_camera, self.choice_ctrl)
        self.Bind(wx.EVT_SLIDER, self.on_seek, self.slider)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_update)

        self.reference_image_path = None
        self.reference_face_img = None
        self.video_path = None
        self.reference_face_descriptor = None

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("data/dlib/shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("data/dlib/dlib_face_recognition_resnet_model_v1.dat")

        self.cap = None
        self.playing = False
        self.tracker = dlib.correlation_tracker()
        self.tracking_face = False
        self.match_found = False
        self.match_start_time = 0
        self.frame_count = 0
        self.detection_interval = 5  # Detect every 5 frames
        self.dist = 1
        self.playback_speed = 4
        self.isUseCamera = False
        self.total_frames = 1
        self.frame_rate = 1

        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Centre()
        self.Show()
        self.choice_ctrl.Show(self.isUseCamera)

    def load_reference_image(self, event):
        openFileDialog = wx.FileDialog(self, "Open Reference Image", "", "",
                                       "Image files (*.jpg;*.png)|*.jpg;*.png",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return

        self.reference_image_path = openFileDialog.GetPath()
        self.image_path_tcl.SetLabel(self.reference_image_path)
        self.load_face_descriptor()
        self.display_reference_image()

    def load_face_descriptor(self):
        if self.reference_image_path:
            self.reference_face_descriptor, self.reference_face_img = self.compute_face_descriptor(
                self.reference_image_path)
            if self.reference_face_descriptor is None:
                wx.MessageBox("Failed to load external face descriptor.", "Error", wx.OK | wx.ICON_ERROR)

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

    def display_reference_image(self):
        if self.reference_face_img is None:
            return
        reference_image_resized = cv2.resize(self.reference_face_img, (128, 128))
        self.update_video_frame(reference_image_resized)

    def load_video(self, event):
        openFileDialog = wx.FileDialog(self, "Open Video", "", "",
                                       "Video files (*.mp4;*.avi)|*.mp4;*.avi",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return
        self.video_path = openFileDialog.GetPath()
        self.video_path_tcl.SetValue(self.video_path)
        self.use_video(self.video_path)

    def use_video(self, video_path):
        if self.isUseCamera:
            if self.choice_ctrl.GetSelection() == 0:
                self.cap = cv2.VideoCapture(0)
            else:
                video_path = self.video_path_tcl.GetValue()
                if validate_input(video_path) is False:
                    wx.MessageBox("Please input right url. support RTSP|HTTP|HTTPS.", "Error", wx.OK | wx.ICON_ERROR)
                    return False
                self.cap = cv2.VideoCapture(video_path)
        else:
            if os.path.exists(video_path) is False:
                return False
            self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.slider.SetRange(0, self.total_frames)
        self.update_video_frame(self.get_frame(0))
        self.on_update(None)
        return True

    def on_play_pause(self, event):
        if not self.cap:  # self.video_path_tcl.GetValue()
            self.use_video(self.video_path_tcl.GetValue())
        if not self.cap:
            wx.MessageBox("Please load the video file first.", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.playing = not self.playing
        self.play_pause_btn.SetLabel("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.Start(1000 // (self.frame_rate * self.playback_speed))
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

    def on_speed_change(self, event):
        if self.playback_speed < 10:
            self.playback_speed = min(self.playback_speed + 1, 10)
        else:
            self.playback_speed = 1

        self.speed_btn.SetLabel(f"Speed x{self.playback_speed}")
        if self.playing:
            self.timer.Start(1000 // (self.frame_rate * self.playback_speed))

    def on_use_camera(self, event):
        checkbox = event.GetEventObject()
        self.isUseCamera = checkbox.IsChecked()
        if self.isUseCamera:
            print("dddd")
        else:
            # 释放所有摄像头
            if self.cap:
                self.cap.release()
                self.cap = None
        # video btn show/hide
        self.load_video_btn.Show(not self.isUseCamera)
        self.choice_ctrl.Show(self.isUseCamera)
        self.video_path_tcl.Show(not self.isUseCamera or (self.isUseCamera and self.choice_ctrl.GetSelection() != 0))

    def on_chioce_camera(self, event):
        self.video_path_tcl.Show(self.isUseCamera and self.choice_ctrl.GetSelection() != 0)
        if self.cap:
            self.cap.release()
            self.cap = None

        self.playing = False
        self.play_pause_btn.SetLabel("Pause" if self.playing else "Play")
        self.timer.Stop()

    def on_seek(self, event):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.slider.GetValue())
            self.playing = False
            self.play_pause_btn.SetLabel("Play")
            self.timer.Stop()
            self.update_time_display(self.slider.GetValue())
            self.update_video_frame(self.get_frame(self.slider.GetValue()))
            self.on_update(None)

    def get_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return None
        # 摄像头左右反了
        if self.isUseCamera:
            frame = cv2.flip(frame, 1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def on_update(self, event):
        if not self.cap:
            return
        if self.reference_face_descriptor is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        # 摄像头左右反了
        if self.isUseCamera:
            frame = cv2.flip(frame, 1)
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
            if self.dist < 0.4:
                pos = self.tracker.get_position()
                pt1 = (int(pos.left()), int(pos.top()))
                pt2 = (int(pos.right()), int(pos.bottom()))
                match_percentage = (1 - self.dist) * 100
                cv2.rectangle(frame_rgb, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"Matched: {match_percentage:.2f}%", (pt1[0], pt1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
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
        if frame is None:
            return
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        panel_w, panel_h = self.video_panel.GetSize().GetWidth(), self.video_panel.GetSize().GetHeight()

        new_w = panel_w
        new_h = int(new_w / aspect_ratio)
        if new_h > panel_h:
            new_h = panel_h
            new_w = int(new_h * aspect_ratio)

        resized_frame = cv2.resize(frame, (new_w, new_h))
        combined_image = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        combined_image.fill(0)

        x_offset = (panel_w - new_w) // 2
        y_offset = (panel_h - new_h) // 2
        combined_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

        # Overlay reference image at the top left corner of the combined image
        reference_image_resized = cv2.resize(self.reference_face_img, (128, 128))
        combined_image[:128, :128] = reference_image_resized

        return combined_image

    def update_video_frame(self, combined_image):
        if combined_image is None:
            return
        height, width = combined_image.shape[:2]
        bitmap = wx.Bitmap.FromBuffer(width, height, combined_image)
        self.video_bitmap.SetBitmap(bitmap)
        self.video_bitmap.Centre()
        self.video_panel.Layout()

    def update_time_display(self, current_frame):
        total_seconds = current_frame // self.frame_rate
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        total_video_seconds = self.total_frames // self.frame_rate
        total_minutes = total_video_seconds // 60
        total_seconds = total_video_seconds % 60
        self.time_display.SetLabel(f"{minutes:02}:{seconds:02} / {total_minutes:02}:{total_seconds:02}")

    def on_resize(self, event):
        # 刷新示图位置
        self.on_update(None)
        self.Layout()
        event.Skip()


if __name__ == "__main__":
    app = wx.App()
    FaceRecognitionApp(None, title='Face Recognition')
    app.MainLoop()
