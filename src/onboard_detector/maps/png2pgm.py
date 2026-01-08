import cv2

def png_to_pgm_cv(png_path, pgm_path):
    # 读取为灰度图像
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    
    # 保存为PGM
    cv2.imwrite(pgm_path, img)
    
    print(f"OpenCV转换成功：{png_path} → {pgm_path}")

# 示例用法
png_to_pgm_cv('map.png', 'map_mt.pgm')
