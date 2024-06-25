
import cv2
import operator
import matplotlib.pyplot as plt
import os
 
def img_input(myMark_path, result_path):
    myMark = cv2.imread(myMark_path)  # 使用 cv2 来打开图片
    result = cv2.imread(result_path)  # 使用 cv2 来打开图片
    return (myMark, result)
 
 
def img_size(img):
    white = 0
    black = 0
    list1 = [255, 255, 255]
    list2 = [0,0,0]
    count = 0
    c=0
    for x in img:
        for y in x:
            count = count + 1
            if operator.eq(y.tolist(), list1)==True:
                white = white + 1
            elif operator.eq(y.tolist(), list2)==True:
                black = black + 1
            else:
                c=c+1
    return (c,count, white,black)
 
def size_same(img1,img2):
    size = 0
    list = [255,255,255]
    for x1,x2 in zip(img1,img2):
        for y1,y2 in zip(x1,x2):
            if operator.eq(y1.tolist(), y2.tolist()) & operator.eq(y1.tolist(), list):
                size = size +1
    return size
 
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    img_path = "/pre/"
    label_path = "/mask/"
    dir = os.listdir(label_path)
    dicelist = []
    for i in dir:
        im = os.path.join(img_path,str(i))
        lab = os.path.join(label_path,str(i))
        
        #print("------",myMark_path)
        #print("------", result_path)
        myMark, result = img_input(lab, im)
        # print("img:", img, "label:", label)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        c1,count1, white1, black1 = img_size(myMark)
        c2,count2, white2, black2 = img_size(result)
        size = size_same(myMark, result)
        dice = 2 * size / (white1 + white2)
        dicelist.append(dice)
        #print("grey1:",c1,"conut1:",count1,"white1:", white1, "black1:", black1)
        #print("grey2:",c2,"conut2:",count2,"white2:", white2, "black2:", black2)
        #print("same size:", size)
        #print("dice:", dice)
        print(dice)
        #print("dicelist:", dicelist)
 
 
    x_values = range(1, 72)
    y_values = dicelist
    '''
    scatter() 
    x:横坐标 y:纵坐标 s:点的尺寸
    '''
    #plt.scatter(x_values, y_values, s=50)
 
    # 设置图表标题并给坐标轴加上标签
    #plt.title('Dice', fontsize=24)
    #plt.xlabel('Image sequence number', fontsize=14)
    #plt.ylabel('dice', fontsize=14)
 
    # 设置刻度标记的大小
    #plt.tick_params(axis='both', which='major', labelsize=14)
    #plt.show()
 
    dicesum = 0
    num = 0
    for num in range(0, 76):
        dicesum = dicesum + dicelist[num]
    print("dicesum:",dicesum,"num:",num)
    avg = dicesum/76
    print("avg:",avg)