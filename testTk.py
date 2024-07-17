import wx


class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # 创建一个 wx.Choice 控件
        choices = ['Option 1', 'Option 2', 'Option 3']
        choice_ctrl = wx.Choice(panel, choices=choices)
        choice_ctrl.SetSelection(0)  # 默认选择第一个选项

        vbox.Add(choice_ctrl, flag=wx.EXPAND | wx.ALL, border=10)

        # 添加一个按钮，用于显示选择的选项
        button = wx.Button(panel, label="Show Selection")
        button.Bind(wx.EVT_BUTTON, lambda event: self.show_selection(choice_ctrl))

        vbox.Add(button, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(vbox)

        self.SetTitle("wx.Choice Example")
        self.SetSize((400, 200))
        self.Centre()

    def show_selection(self, choice_ctrl):
        selection = choice_ctrl.GetString(choice_ctrl.GetSelection())
        wx.MessageBox(f"You selected: {selection}", "Selection", wx.OK | wx.ICON_INFORMATION)


class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None)
        frame.Show(True)
        return True


if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()