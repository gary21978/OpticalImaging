#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的01二值PNG图像转换为0-255灰度图像脚本
"""

import cv2
import numpy as np
import os


def expand_image_with_padding(img, padding_pixels, fill_value=255):
    """
    将图像外扩n个像素，用指定值填充
    
    参数:
    img: 输入图像（numpy数组）
    padding_pixels: 外扩的像素数
    fill_value: 填充值，默认255（白色）
    
    返回:
    expanded_img: 外扩后的图像
    """
    if len(img.shape) == 2:  # 灰度图
        # 使用cv2.copyMakeBorder进行外扩
        expanded_img = cv2.copyMakeBorder(
            img, 
            padding_pixels,  # top
            padding_pixels,  # bottom  
            padding_pixels,  # left
            padding_pixels,  # right
            cv2.BORDER_CONSTANT,
            value=fill_value
        )
    else:  # 彩色图
        expanded_img = cv2.copyMakeBorder(
            img,
            padding_pixels,
            padding_pixels, 
            padding_pixels,
            padding_pixels,
            cv2.BORDER_CONSTANT,
            value=[fill_value, fill_value, fill_value]
        )
    
    return expanded_img


def expand_image_file(input_path, output_path=None, padding_pixels=10, fill_value=255):
    """
    读取图像文件，外扩后保存
    
    参数:
    input_path: 输入图像文件路径
    output_path: 输出图像文件路径（可选）
    padding_pixels: 外扩的像素数，默认10
    fill_value: 填充值，默认255（白色）
    
    返回:
    bool: 操作是否成功
    """
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_expanded{ext}"
    
    try:
        # 读取图像
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"错误: 无法读取图像 {input_path}")
            return False
        
        # 获取原始尺寸
        if len(img.shape) == 2:
            height, width = img.shape
            print(f"原始图像尺寸: {width}x{height} (灰度图)")
        else:
            height, width, channels = img.shape
            print(f"原始图像尺寸: {width}x{height}x{channels} (彩色图)")
        
        # 外扩图像
        expanded_img = expand_image_with_padding(img, padding_pixels, fill_value)
        
        # 获取新尺寸
        if len(expanded_img.shape) == 2:
            new_height, new_width = expanded_img.shape
        else:
            new_height, new_width = expanded_img.shape[:2]
        
        print(f"外扩后尺寸: {new_width}x{new_height}")
        print(f"外扩像素数: {padding_pixels}, 填充值: {fill_value}")
        
        # 保存图像
        success = cv2.imwrite(output_path, expanded_img)
        if success:
            print(f"图像外扩完成: {output_path}")
            return True
        else:
            print(f"错误: 保存图像失败 {output_path}")
            return False
            
    except Exception as e:
        print(f"图像外扩失败: {e}")
        return False


def convert_binary_png(input_path, output_path=None, crop_rect=None, expand_pixels=None, fill_value=255):
    """
    将01二值PNG图像转换为0-255灰度图像，并可选择性裁剪和外扩

    参数:
    input_path: 输入PNG文件路径
    output_path: 输出PNG文件路径（可选，默认在原文件名后加_converted）
    crop_rect: 裁剪矩形 (x1, y1, x2, y2)，其中(x1,y1)是左上角，(x2,y2)是右下角
              如果为None，则不进行裁剪
    expand_pixels: 外扩像素数，如果为None则不外扩
    fill_value: 外扩时的填充值，默认255
    """
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_converted{ext}"

    try:
        # 读取图像（灰度模式）
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"错误: 无法读取图像 {input_path}")
            return False

        # 检查图像的像素值范围
        unique_values = np.unique(img)
        print(f"原图像像素值: {unique_values}")

        # 如果是0和1的二值图像，转换为0和255
        if len(unique_values) == 2 and np.array_equal(unique_values, [0, 1]):
            img_converted = img * 255
            print("转换: 0,1 -> 0,255")
        elif len(unique_values) == 2 and np.array_equal(unique_values, [0, 255]):
            img_converted = img
            print("图像已经是0-255格式，无需转换")
        else:
            # 对于其他情况，使用阈值处理
            print("使用阈值处理转换为0-255二值图像")
            _, img_converted = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 裁剪图像：根据指定的矩形区域裁剪
        height, width = img_converted.shape
        
        if crop_rect is not None:
            x1, y1, x2, y2 = crop_rect
            
            # 验证裁剪坐标的有效性
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
                print(f"警告: 裁剪坐标无效 ({x1}, {y1}, {x2}, {y2})，图像尺寸为 {width}x{height}")
                print("保持原图像不进行裁剪")
                img_final = img_converted
            else:
                # 使用NumPy切片进行裁剪 [y1:y2, x1:x2]
                img_final = img_converted[y1:y2, x1:x2]
                crop_width = x2 - x1
                crop_height = y2 - y1
                print(f"图像裁剪成功:")
                print(f"  原尺寸: {width}x{height}")
                print(f"  裁剪区域: ({x1}, {y1}) 到 ({x2}, {y2})")
                print(f"  新尺寸: {crop_width}x{crop_height}")
        else:
            img_final = img_converted
            print("未指定裁剪区域，保持原图像尺寸")

        # 外扩图像（如果需要）
        if expand_pixels is not None and expand_pixels > 0:
            img_final = expand_image_with_padding(img_final, expand_pixels, fill_value)
            final_height, final_width = img_final.shape
            print(f"图像外扩: 外扩{expand_pixels}像素，填充值{fill_value}")
            print(f"最终尺寸: {final_width}x{final_height}")

        # 保存转换后的图像
        cv2.imwrite(output_path, img_final)
        print(f"转换完成: {output_path}")

        # 验证结果
        result_unique = np.unique(img_final)
        print(f"转换后像素值: {result_unique}")

        return True

    except Exception as e:
        print(f"转换失败: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 示例：转换单个文件
    input_path = "./DSW-c.png"  # 替换为你的输入文件路径

    # 方法1：不裁剪，只转换
    # convert_binary_png(input_path)

    # 方法2：指定裁剪区域 (x1, y1, x2, y2)
    # 例如：裁剪从(100, 50)到(400, 300)的矩形区域
    # convert_binary_png(input_path, crop_rect=(100, 50, 400, 300))
    
    # 方法3：外扩图像
    # 例如：外扩20个像素，填充值为255
    # convert_binary_png(input_path, expand_pixels=17, fill_value=255)
    
    # 方法4：组合操作（先裁剪后外扩）
    convert_binary_png(input_path, crop_rect=(52, 0, 52+366, 366), expand_pixels=17, fill_value=255)
    
    # 方法5：只外扩图像文件
    # expand_image_file(input_path, padding_pixels=20, fill_value=255)

    exit()
    # 交互式使用
    if os.path.exists(input_path):
        print(f"处理图像: {input_path}")
        
        # 读取图像获取尺寸信息
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            height, width = img.shape
            print(f"图像尺寸: {width}x{height}")
            
            # 询问是否需要裁剪
            crop_choice = input("是否需要裁剪图像？(y/n): ").strip().lower()
            
            if crop_choice == 'y':
                print("请输入裁剪区域的坐标:")
                print("左上角坐标 (x1, y1):")
                try:
                    x1 = int(input("  x1: "))
                    y1 = int(input("  y1: "))
                    print("右下角坐标 (x2, y2):")
                    x2 = int(input("  x2: "))
                    y2 = int(input("  y2: "))
                    
                    crop_rect = (x1, y1, x2, y2)
                    print(f"设置裁剪区域: {crop_rect}")
                    convert_binary_png(input_path, crop_rect=crop_rect)
                    
                except ValueError:
                    print("输入格式错误，使用默认处理（不裁剪）")
                    convert_binary_png(input_path)
            else:
                convert_binary_png(input_path)
        else:
            print("无法读取图像文件")
    else:
        print("文件不存在!")
        
        # 如果默认文件不存在，提供交互式输入
        input_path = input("请输入PNG文件路径: ").strip()
        if os.path.exists(input_path):
            # 读取图像获取尺寸信息
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                height, width = img.shape
                print(f"图像尺寸: {width}x{height}")
                
                # 询问是否需要裁剪
                crop_rect = None
                expand_pixels = None
                fill_value = 255
                
                # 询问裁剪选项
                crop_choice = input("是否需要裁剪图像？(y/n): ").strip().lower()
                if crop_choice == 'y':
                    print("请输入裁剪区域的坐标:")
                    print("左上角坐标 (x1, y1):")
                    try:
                        x1 = int(input("  x1: "))
                        y1 = int(input("  y1: "))
                        print("右下角坐标 (x2, y2):")
                        x2 = int(input("  x2: "))
                        y2 = int(input("  y2: "))
                        crop_rect = (x1, y1, x2, y2)
                        print(f"设置裁剪区域: {crop_rect}")
                    except ValueError:
                        print("输入格式错误，将跳过裁剪")
                        crop_rect = None
                
                # 询问外扩选项
                expand_choice = input("是否需要外扩图像？(y/n): ").strip().lower()
                if expand_choice == 'y':
                    try:
                        expand_pixels = int(input("请输入外扩的像素数: "))
                        fill_input = input("请输入填充值 (默认255): ").strip()
                        if fill_input:
                            fill_value = int(fill_input)
                        print(f"设置外扩: {expand_pixels}像素，填充值: {fill_value}")
                    except ValueError:
                        print("输入格式错误，将跳过外扩")
                        expand_pixels = None
                
                # 执行转换
                convert_binary_png(input_path, crop_rect=crop_rect, expand_pixels=expand_pixels, fill_value=fill_value)
            else:
                print("无法读取图像文件")
        else:
            print("文件不存在!")