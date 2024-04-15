import os
import imageio

# 定义文件夹路径
folder_path = r'D:\CS5340\CS5340-Project\result\mrf_e_t_s_3135\dogs-jump'

images = []

# 读取文件夹内所有png文件，并添加到列表中
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith('.png'):
        file_path = os.path.join(folder_path, file_name)
        images.append(imageio.imread(file_path))

# 计算每帧的持续时间，使总时长约为10秒
total_duration = 10  # 总时长为10秒
frame_duration = total_duration / len(images)  # 计算每帧持续时间

# 保存这些图片为一个GIF文件，设置每帧持续时间
output_path = os.path.join(folder_path, 'output.gif')
imageio.mimsave(output_path, images, duration=frame_duration)
