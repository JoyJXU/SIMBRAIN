import os
import platform
import torch
import time
import datetime

def gpu_memory():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    used_memory = 0
    # print(gpu_status)
    for device_id in range(torch.cuda.device_count()):
        used_memory += int(gpu_status[2 + 4 * device_id].split('/')[0].split('M')[0].strip())
    return used_memory


if __name__ == '__main__':
    if platform.system() == 'Linux':
        print("Enable auto shutdown function")
        for _ in range(10):
            time.sleep(10)
            if gpu_memory() < 50:
                time.sleep(10)
                if gpu_memory() < 50:
                    print('\nGPU free, shutdown')
                    os.system("shutdown")
            else:
                print("\r", datetime.datetime.now().strftime("%m-%d %H:%M"), "GPU occupied...", end="")
    else:
        print("auto shutdown only support Linux platform")
