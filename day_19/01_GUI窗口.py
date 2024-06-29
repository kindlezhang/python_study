"""
简单的视窗GUI界面
tkinter
"""
import tkinter as tk

# 创建窗口
window = tk.Tk()  # 窗口对象
window.title("my window")  # 名称
window.geometry("400x200")  # 长乘宽，字母x

var = tk.StringVar()  # tk中特有的字符串形式

l = tk.Label(window, textvariable=var,  # 直接回车就可以换行，链式调用的书写换行需要用\
             bg="green", font=("Arial", 12), width=15,
             height=2)  # 标签对象，或者text="文本内容"
l.pack()  # place或者pack进行安置

on_hit = False


def hit_me():
    global on_hit
    if not on_hit:  # 用not
        on_hit = True
        var.set("you hit me")
    else:
        on_hit = False
        var.set('')


b = tk.Button(window, text="hit me", width=15,  # 生成按钮对象
              height=2, command=hit_me)
b.pack()  # 放置按钮

window.mainloop()
