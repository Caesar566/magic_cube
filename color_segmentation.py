import cv2 as cv
import numpy as np
import kociemba 

#在上面展开方式的基础上，用U、R、F、D、L、B六个字母分别表示六个面中心对应的颜色，
#将魔方各个色块的颜色按照U1、U2…顺序，字母顺序为U、R、F、D、L、B编号,kociemba.solve('')
#U、yellow
#R、green
#F、red
#D、white
#L、blue
#B、orange

color = ['yellow', 'white', 'red', 'orange', 'blue' ,'green']
#颜色的数据，两个列表的量必须一一对应，避免后面的循环出现bug
low_color_green = ([0, 150, 0])#各颜色BGR阈值
high_color_green = ([100, 255, 100])
low_color_yellow = ([0, 150, 150])
high_color_yellow = ([120, 255, 255])
low_color_blue = ([100, 0, 0])
high_color_blue = ([255, 100, 100])
low_color_white = ([100, 100, 100])
high_color_white = ([255, 255, 255])
low_color_orange = ([0 ,0, 150])
high_color_orange = ([150 ,150, 255])
low_color_red = ([0, 0, 30])
high_color_red = ([30, 30, 255])

color_green = [low_color_green, high_color_green]#将颜色阈值最大值与最小值打包
color_blue = [low_color_blue, high_color_blue]
color_yellow = [low_color_yellow, high_color_yellow]
color_white = [low_color_white, high_color_white]
color_red = [low_color_red, high_color_red]
color_orange = [low_color_orange, high_color_orange]

color_data = [color_yellow, color_white, color_red, color_orange, color_blue, color_green]#将阈值最大值与最小值打包，方便复用

#打包，便于利用

def pic_show(window_name, img_pro):#Display the processed image
    cv.imshow(window_name, img_pro)
    cv.waitKey(0)

def Erosion_Dilation(img):#Corrosion and dilation of picture
    kernel = np.ones((20, 20))
    erosion = cv.erode(img, kernel, iterations=1)
    erosion_dilation = cv.dilate(erosion, kernel, iterations=1) 
    return erosion_dilation
    
def remove_zero(list_1):
    n_0 = 0
    for k in list_1 [::-1]:
        if k == 0:
            n_0 += 1
        else:
            break
    list_1 = list_1 [:-n_0]
    return list_1

def img_seg(img_pro):#Color segmentation of the picture
    i = 0#定义方块的列表的下标，作为位移
    block_c1 = 0#存放每层的值，用来做归一化
    block_c2 = 0
    block_c3 = 0

    yellow = [0 for i in range(9)]#设置各颜色的列表，存储识别后颜色块的坐标
    white = [0 for i in range(9)]
    red = [0 for i in range(9)]
    orange = [0 for i in range(9)]
    blue = [0 for i in range(9)]
    green= [0 for i in range(9)]
    color_location = [yellow, white, red, orange, blue, green]#打包

    all_location = [0 for i in range(9)]
    for color_ID in range(len(color)):
        j = 0#定义一个循环变量，作为列表的位移，一种颜色的坐标的列表的位移
        mask = cv.inRange(img_pro, np.array(color_data[color_ID][0]), np.array(color_data[color_ID][1]))
        #judge color，将图片进行二值化的处理，将在阈值中的颜色区域标白
        mask = Erosion_Dilation(mask)#将标白后的二值化的图片进行腐蚀膨胀处理
        pic_show(color[color_ID], mask)#将处理后的二值化的图片进行显示
        print(str(color[color_ID]+" is  "))
        size, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        #去分割颜色区域，得出颜色坐标放入numpy库中的坐标
        if size != ():#如果得出的坐标不为空的话，即在这张图像中有这种颜色
            for cnt in size:#从坐标中拿出变量
                (x, y, w, h) = cv.boundingRect(cnt)#将得出的多个坐标取出，得出一个矩形框坐标      
                if w - x >100:#只为白色准备，因为当背景为白色，会将白色进行误判，所以设置阈值，进行筛选
                    pass  
                else:#同一行横坐标统一
                    if i == 0:
                        #第一个坐标进入时，将坐标保留和接下来的坐标对比，如果在相近的范围内，就将两个值归一化
                        #如果不同，就往下放入第二个值，因为只有三层，所以需要进行三层比较
                        block_c1 = y
                    else:
                        if y + 10 >= block_c1 and y - 10 <= block_c1:
                            y = block_c1
                        else:
                            if block_c2 == 0:
                                block_c2 = y
                            else:
                                if y + 10 >= block_c2 and y - 10 <= block_c2:
                                    y = block_c2
                                else:
                                    if block_c3 == 0:
                                        block_c3 = x
                                    else:
                                        if y + 10 >= block_c3 and y - 10 <= block_c3:
                                            y = block_c3
                    print(x, y)
                    if i == 9:#有可能会出现方块齐全的情况，当再进行i += 1 时，会超出列表的范围。
                        break
                    all_location[i] = x, y#将坐标的x, y放入存放所有坐标的列表中
                    all_location[i] = list(all_location[i])#将元组转化为列表
                    color_location[color_ID][j] = x, y#将坐标按颜色分类放入列表中
                    color_location[color_ID][j] = list(color_location[color_ID][j])
                    j+=1
                    i+=1              
        else:   
            print('None') 
    for i in range(len(color)):
        color_location[i] = remove_zero(color_location[i])#去除没有必要的0
    return color_location, all_location#color_loction:返回存颜色与坐标对应的列表
                                       #all_location:存放乱序的的坐标的列表
    
def data_processing(location_list, all_location):#进行数据处理与数据的比较
    end_str = []
    floor_1 = [0 for i in range(3)]
    floor_2 = [0 for i in range(3)]
    floor_3 = [0 for i in range(3)]

    print(location_list)
    print(all_location)

    for a in range(len(all_location)):#将列表按y轴进行排序
        for b in range(len(all_location)-a-1):
            if all_location[b][1] > all_location[b+1][1]:
                all_location[b+1], all_location[b] = all_location[b], all_location[b+1]
    print(all_location)

    for a in range(3):#将列表分成三层，即代表魔方的每层
        floor_1[a] = all_location[a]
    for a in range(3):
        floor_2[a] = all_location[a + 3]
    for a in range(3):
        floor_3[a] = all_location[a + 6]

    for a in range(len(floor_1)):#将分出来的每层，按x轴坐标排序
        for b in range(len(floor_1)-a-1):
            if floor_1[b][0] > floor_1[b+1][0]:
                floor_1[b+1], floor_1[b] = floor_1[b], floor_1[b+1]  
    for a in range(len(floor_2)):
        for b in range(len(floor_2)-a-1):
            if floor_2[b][0] > floor_2[b+1][0]:
                floor_2[b+1], floor_2[b] = floor_2[b], floor_2[b+1]
    for a in range(len(floor_3)):
        for b in range(len(floor_3)-a-1):
            if floor_3[b][0] > floor_3[b+1][0]:
                floor_3[b+1], floor_3[b] = floor_3[b], floor_3[b+1]
    print(floor_1)
    print(floor_2)
    print(floor_3)

    for i in range(len(floor_1)):#将排序好的all_location列表覆盖之前乱序的all_location列表进行
        all_location[i] =  floor_1[i]      
    for i in range(len(floor_2)):
        all_location[i + 3] =  floor_2[i]
    for i in range(len(floor_1)):
        all_location[i + 6] =  floor_3[i]
    print(all_location)

    for a in range(9):#将每块的坐标与之前的color_location进行对比，得出每块坐标对应的颜色，生成一个颜色的列表
        for b in range(len(location_list[0])):
            if all_location[a] == location_list[0][b]:
                end_str.append('U') 
        for b in range(len(location_list[1])):
            if all_location[a] == location_list[1][b]:
                end_str.append('D')
        for b in range(len(location_list[2])):
            if all_location[a] == location_list[2][b]:
                end_str.append('F')
        for b in range(len(location_list[3])):
            if all_location[a] == location_list[3][b]:
                end_str.append('B')
        for b in range(len(location_list[4])):
            if all_location[a] == location_list[4][b]:
                end_str.append('L')
        for b in range(len(location_list[5])):
            if all_location[a] == location_list[5][b]:
                end_str.append('R')
    print(end_str)
    return end_str#返回颜色的列表

 
def check_str(block):#检查每个颜色是否出现了9次，即验证颜色识别的准确性
    num = 0
    U = 0
    D = 0
    B = 0
    F = 0
    R = 0
    L = 0
    cube_list = [U, B, D, F, R, F, L]#两个列表必须对应
    cube_color = ['U', 'B', 'D', 'F', 'R', 'F', 'L']
    for i in range(len(cube_color)):#
          for j in range(6):
            for k in range(9):
                if block[j][k] == cube_color[i]:
                    cube_list[i] += 1
    for i in range(len(cube_list)):
        if cube_list[i] == 9:
            num += 1
        else:
            print(cube_color[i] + '数量错误 ,数量为'+ str(cube_list[i]))
    if num == 6:
        return 1
    else:
        return 0

def main():
    block = []
    str_list = [0 for i in range(9)]
    for i in range(6):#循环6次，拍摄魔方6个面
        pic_path = 'D:\magic cube\cube_pic\\' + str(i) + '.jpg'   #the color are distributed according to BGR
        img = cv.imread(pic_path)#读取图片
        img_pro = cv.resize(img, (500, 500))#设置图片大小
        img_pro = cv.GaussianBlur(img_pro, (5, 5), 0 )#模糊化
        img_pro = Erosion_Dilation(img_pro) #进行腐蚀膨胀
        pic_show('image', img_pro)#将图像处理结果显示
        location_list, all_location = img_seg(img_pro)#用二值化的方式，将图像进行分割
        end_str_list = data_processing(location_list, all_location)#处理列表数据，排列
        str_list[i] = ''.join(end_str_list)#将列表转化为字符串
        if block == 0:
            block = end_str_list
        else:
            block.append(end_str_list)#生成一个关于坐标的列表
         
    end_str = str_list[0] + str_list[1] + str_list[2] + str_list[3] + str_list[4] + str_list[5]
    #生成一个6面完整的字符串
    print(end_str)
    check_back = check_str(block)#去检查每块的数量是否正确
    if check_back == 1:#正确，放入解魔方的函数，进行求解
        kociemba.solve(end_str)
    else:
        print('识别错误')        

if __name__ == '__main__':    
    main()