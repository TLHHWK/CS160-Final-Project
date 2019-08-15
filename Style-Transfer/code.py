# -*- coding: utf-8 -*-

## 载入所需库
import cv2
import time
import sys
import django
from PIL import Image


def style_transfer(pathIn='',
                   pathOut='',
                   model='',
                   width=None,
                   jpg_quality=80):
    '''
    pathIn: 原始图片的路径address of the orignial picture
    pathOut: 风格化图片的保存路径address of the modified picture
    model: 预训练模型的路径 the adress of the model
    width: 设置风格化图片的宽度，默认为None, 即原始图片尺寸//
    jpg_quality: 0-100，设置输出图片的质量，默认80，越大图片质量越好
    '''
    
    ## 读入原始图片，调整图片至所需尺寸，然后获取图片的宽度和高度
    img = cv2.imread(pathIn)
    (h, w) = img.shape[:2]
    if width is not None:
        img = cv2.resize(img, (width, round(width*h/w)), interpolation=cv2.INTER_CUBIC)
        (h, w) = img.shape[:2]

    ## 从本地加载预训练模型
    print('Loading the trained models......')
    net = cv2.dnn.readNetFromTorch(model)
    
    ## 将图片构建成一个blob：设置图片尺寸，将各通道像素值减去平均值（比如ImageNet所有训练样本各通道统计平均值）
    ## 然后执行一次前馈网络计算，并输出计算所需的时间
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()
    print("Time for Style Transfer：{:.2f}秒".format(end - start))

    ## reshape输出结果, 将减去的平均值加回来，并交换各颜色通道
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output = output.transpose(1, 2, 0)
    
    ## 输出风格化后的图片
    cv2.imwrite(pathOut, output, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    

   
## 测试
import glob
models = glob.glob('D:/BerkeleyFinal/Style-Transfer/*/*/*.t7')
# models = glob.glob('./*/*/*.t7')

models      ## 列出所有可用的预训练模型
#print(models)
#print(len(models))
print(sys.argv[1])

targetIndex = int(sys.argv[1])
targetType = sys.argv[2]
targetName = sys.argv[3]

# pathIn = '../PhoneCaseGenerator/DoubleLift.jpg'
# pathOut = '../PhoneCaseGenerator/DoubleLiftFinal.jpg'
# model = models[targetIndex]

if targetType =="CLIP":
    pathIn = 'C:/Users/HWK/Downloads/'+ targetName+'.png'
    pathOut = 'C:/Users/HWK/Downloads/'+ targetName+'Result.png'
    model=models[targetIndex]
    style_transfer(pathIn, pathOut, model)

if targetType =="ALL":
    pathIn = 'C:/Users/HWK/Downloads/'+ targetName+'.png'
    pathOut = 'C:/Users/HWK/Downloads/'+ targetName+'Result.png'
    model=models[targetIndex]
    style_transfer(pathIn, pathOut, model)



# pathIn = 'C:/Users/HWK/Downloads/t.png'
# pathOut = 'C:/Users/HWK/Downloads/testFinal.jpg'
# model = models[targetIndex]





# pathIn = './img/timg.jpg'
# pathOut = './result/result_imghwk.jpg'
# model = './models/instance_norm/udnie.t7'
# style_transfer(pathIn, pathOut, model, width=500)


# pathIn = './img/img02.jpg'
# pathOut = './result/result_img02.jpg'
# model = './models/instance_norm/starry_night.t7'
# style_transfer(pathIn, pathOut, model, width=500)


# pathIn = 'D:/BerkeleyFinal/Style-Transfer/img/img03.jpg'
# pathOut = 'D:/BerkeleyFinal/Style-Transfer/result/result_img03.jpg'
# model = 'D:/BerkeleyFinal/Style-Transfer/models/instance_norm/the_scream.t7'
# style_transfer(pathIn, pathOut, model, width=500)


#fail
# pathIn = './img/img04.jpg'
# pathOut = './result/result_img04.jpg'
# model = './models/eccv16/the_wave.t7'
# style_transfer(pathIn, pathOut, model, width=500)
#
#fail
# pathIn = './img/img05.jpg'
# pathOut = './result/result_img05.jpg'
# model = './models/instance_norm/mosaic.t7'
# style_transfer(pathIn, pathOut, model, width=500)

# pathIn = 'D:/BerkeleyFinal/PhoneCaseGenerator/DoubleLift.jpg'
# pathOut = 'D:/BerkeleyFinal/PhoneCaseGenerator/DoubleLiftFinal.jpg'
# model = 'D:/BerkeleyFinal/Style-Transfer/models/eccv16/composition_vii.t7'
# style_transfer(pathIn, pathOut, model)
# img = Image.open(pathOut)
# img.show()