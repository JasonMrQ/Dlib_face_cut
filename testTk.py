import wx


class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent, title=title, size=(800, 600))

        # 设置窗口的初始大小
        self.SetSize(800, 600)

        # 设置窗口的最小和最大大小
        self.SetMinSize((400, 300))
        self.SetMaxSize((1024, 768))

        self.Centre()  # 窗口居中

        # 添加一个简单的面板和按钮
        panel = wx.Panel(self)
        button = wx.Button(panel, label="点击我", pos=(350, 250))

        self.Bind(wx.EVT_BUTTON, self.on_button_click, button)




    def on_button_click(self, event):
        # wx.MessageBox("按钮被点击了！", "提示", wx.OK | wx.ICON_INFORMATION)
        wx.adv.CreateFileTipProvider()


if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame(None, "wxPython 示例")
    frame.Show(True)
    app.MainLoop()