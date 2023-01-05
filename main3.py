import numpy as np
import cv2
import os
from scipy.ndimage import rotate

dims = 4400


# hàm chuyển data về dạng số nguyên
def convert2Int(x):
    noise = 0
    noise = np.random.randint(-3, 3)
    x_ = x.split()
    a = int(float(x_[0]) * 1000) + noise
    b = int(float(x_[1]) * 1000) + noise
    return a, b


# hàm tính khoảng cách giữa 2 điểm
def khoang_cach(x1, y1, x2, y2):
    return np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2))


# hàm để vẽ đường thẳng từ đường thẳng
def show_line(out_image, line):
    x_ = line[0]
    y_ = line[1]
    angle = line[2]
    r_ = line[3]
    x0 = int(x_ - r_ * np.cos(angle * np.pi / 180))
    x1 = int(x_ + r_ * np.cos(angle * np.pi / 180))
    y0 = int(y_ - r_ * np.sin(angle * np.pi / 180))
    y1 = int(y_ + r_ * np.sin(angle * np.pi / 180))
    out_image = cv2.line(out_image, (x0, y0), (x1, y1), (255, 255, 255), 4, cv2.LINE_AA)
    return out_image


# hàm chính
def detectVL(
        data,
        _show=False,  # True : hiển thị ảnh
        _test=False,  # True : hiển thị accuracy nếu là file test
        # >>> trả về x, y, angle của VL marker, nếu ko có trả về None, None, None <<<
):
    B3_image = np.zeros((dims, dims, 3), np.uint8)
    B3_2_3_image = np.zeros((dims, dims, 3), np.uint8)
    blank_image = np.zeros((dims, dims, 3), np.uint8)
    out_image = np.zeros((dims, dims, 3), np.uint8)
    matrix = np.zeros((dims, dims), dtype=np.uint8)

    if _test == True:
        label = data[0]
        x_ = label
        x_ = label.split()  # nếu truyền ma trận thì xóa
        y_const = x_[0]
        x_const = x_[1]
        # print("label: ", x_const, y_const, x_[2])

    for i in data[1:]:
        a, b = convert2Int(i)
        a += int(dims / 2)
        b += int(dims / 2)
        if a >= dims: a = dims - 1
        if b >= dims: b = dims - 1
        matrix[a][b] = 255
    # ///////////////////////////
    # B3, 3.1 Detect line, check trong từng cụm có tồn tại marker ko
    # ///////////////////////////

    # tìm ra các đường thẳng trong data bằng opencv
    linesP = cv2.HoughLinesP(
        matrix,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=110,
        # Min allowed length of line      # hằng số để   kiểm tra các đường thỏa điều kiện có thể là VL marker
        maxLineGap=100  # Max allowed gap between line for joining them
    )
    list_lines = []

    # tính toán ra hệ số góc a và hằng số b của từng đường thẳng
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(blank_image, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 10, cv2.LINE_AA)
            parameters = np.polyfit((l[0], l[2] + 1), (l[1], l[3] + 1), 1)
            angle = np.arctan(parameters[0])
            # if angle<0: angle += np.pi
            angle = angle * 180 / np.pi
            l = list(l)
            l.append(angle)
            list_lines.append(l)

    list_lines_new = []

    for line in list_lines:
        check = False
        angle_old = line[4]
        x_ = (line[0] + line[2]) / 2
        y_ = (line[1] + line[3]) / 2
        angle = line[4]
        r0 = abs(x_ - line[0]) / abs(np.cos(angle * np.pi / 180))
        list_lines_new.append([x_, y_, angle, r0])
        B3_image = show_line(B3_image, [x_, y_, angle, r0])
    # ////////////////////////////
    # B3.2,3 tính giá trị góc và khoảng cách tâm giữa các đường phù hợp với VL marker
    # ///////////////////////////

    list_lines_out = []
    # góc
    agl_const1 = 30
    agl_const2 = 60
    agl_const1_ = 150
    agl_const2_ = 120
    error_angle = 4

    # khoảng cách tâm
    dt_1 = 243
    dt_2 = 130
    error_dt = 20
    for line1 in list_lines_new:
        agl1 = line1[2]
        x1 = line1[0]
        y1 = line1[1]
        r1 = line1[3]
        for line2 in list_lines_new:
            agl2 = line2[2]
            x2 = line2[0]
            y2 = line2[1]
            r2 = line2[3]
            for line3 in list_lines_new:
                agl3 = line3[2]
                x3 = line3[0]
                y3 = line3[1]
                r3 = line3[3]

                # so sánh góc và khoảng cách của các đường thẳng để kiểm tra các đường thẳng đó có thuộc VL marker không
                if ((((agl_const1 - error_angle) <= abs(agl1 - agl2) <= (agl_const1 + error_angle))) or (
                ((agl_const1_ - error_angle) <= abs(agl1 - agl2) <= (agl_const1_ + error_angle)))) and (
                        ((agl_const2 - error_angle) <= abs(agl2 - agl3) <= (agl_const2 + error_angle)) or (
                ((agl_const2_ - error_angle) <= abs(agl2 - agl3) <= (agl_const2_ + error_angle)))):
                    if ((dt_1 - error_dt) <= khoang_cach(x1, y1, x2, y2) <= (dt_1 + error_dt)) and (
                            (dt_2 - error_dt) <= khoang_cach(x2, y2, x3, y3) <= (dt_2 + error_dt)):
                        list_lines_out.append([line1, line2, line3])
                        B3_2_3_image = show_line(B3_2_3_image, line1)
                        B3_2_3_image = show_line(B3_2_3_image, line2)
                        B3_2_3_image = show_line(B3_2_3_image, line3)

                        # ///////////////////////////
    # B4 lọc nhiễu, tính toán kết quả của x, y, angle
    # ///////////////////////////
    if len(list_lines_out):
        list_lines_out = np.average(list_lines_out, axis=0)
        check = True
        for line in list_lines_out:
            x_ = line[0]
            y_ = line[1]
            angle = line[2]
            r_ = line[3]
            x0 = int(x_ - r_ * np.cos(angle * np.pi / 180))
            x1 = int(x_ + r_ * np.cos(angle * np.pi / 180))
            y0 = int(y_ - r_ * np.sin(angle * np.pi / 180))
            y1 = int(y_ + r_ * np.sin(angle * np.pi / 180))
            out_image = cv2.circle(out_image, (int(x_), int(y_)), 20, (255, 0, 0), thickness=10)
            cv2.line(out_image, (x0, y0), (x1, y1), (255, 255, 255), 3, cv2.LINE_AA)
            if check == True:
                x1_out = (x1 - dims / 2) / 1000
                y1_out = (y1 - dims / 2) / 1000
                angle_out = np.tan(angle * np.pi / 180)
                print("out: ", x1_out, y1_out, - angle_out)
                if _test == True:
                    print("acc: ", (1 - abs(x1_out - float(x_const)) / float(x_const)))
            check = False
        # ///////////////////////////
        # hiển thị kết quả bằng hình ảnh
        # ///////////////////////////
        if _show == True:
            B3_image = cv2.resize(B3_image, (600, 600))
            B3_2_3_image = cv2.resize(B3_2_3_image, (600, 600))
            blank_image = cv2.resize(blank_image, (600, 600))
            matrix = cv2.resize(matrix, (600, 600))
            out_image = cv2.resize(out_image, (600, 600))
            print("/////////////")
            cv2.imshow("B3.1", B3_image)
            cv2.imshow("B3.2,3", B3_2_3_image)
            cv2.imshow("B4", out_image)
            cv2.imshow("origin", matrix)
            cv2.waitKey(500)
        return x1_out, y1_out, angle_out
    else:
        return None, None, None


import rospy
from std_msgs.msg import String


def vlMarker():
    pub = rospy.Publisher('data', String, queue_size=10)
    rospy.init_node('vlMarkerNode', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    x1_out, y1_out, angle_out = detectVL(data)
    s = str(x1_out) + " " + str(y1_out) + " " + str(angle_out)

    pub.publish(s)
    # rospy.Subscriber("vl_data", String, callback)
    rate.sleep()


if __name__ == "__main__":
    path = "/home/xuanhieu/data_marker/data/vl/"
    for i in os.listdir(path):
        f = open(path + i)
        data = f.readlines()
        detectVL(data, _show=True, _test=True)
        vlMarker()
        f.close()
pass
