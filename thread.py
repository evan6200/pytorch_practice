from threading import Thread
import time

def timer(name,delay,times):
    print("計時器: "+ name + "開始")
    while times > 0:
        time.sleep(delay)
        print(name + ": " + str(time.ctime(time.time())))
        times -= 1
    print("計時器: " + name + "完成")

def Main():
    t1 = Thread(target=timer,args=("程式1",1,5))
    t2 = Thread(target=timer,args=("程式2",2,5))
    #程式開始
    t1.start()
    t2.start()
    print("\n程式開始")
    #程式結束
    t1.join() # join() 等待程式自然結束或拋出Error
    t2.join()
    print("\n程式結束")


Main()
