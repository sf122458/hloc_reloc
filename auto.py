import os
import time


for x in range(-30, 30, 5):
    for z in range(10, 40, 5):
        print(f"Set position to ({x}, 0, {z})")
        os.system(f"sh shfiles/set_pos.sh {x} 0 {z}")
        time.sleep(10)
        print("Take photo")
        os.system("sh shfiles/take_photo.sh")
        time.sleep(10)