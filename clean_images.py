import os
from PIL import Image


def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im,
                 ((final_size - new_image_size[0]) // 2,
                  (final_size - new_image_size[1]) // 2))
    return new_im


def clean_image_data(input_folder, output_folder, final_size=512):
    """
    Clean the image dataset by resizing and ensuring consistent number of channels.
    :param input_folder: (str) Path to the folder containing the original images.
    :param output_folder: (str) Path to the folder where cleaned images will be saved.
    :param final_size: (int) Desired size for the cleaned images (final_size x final_size).
    :return:
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dirs = os.listdir(input_folder)
    for item in dirs:
        if item.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            try:
                img_path = os.path.join(input_folder, item)
                img = Image.open(img_path)

                # Resize image and ensure it's RGB
                new_im = resize_image(final_size, img)

                # Save the cleaned image
                cleaned_img_path = os.path.join(output_folder, item)
                new_im.save(cleaned_img_path)
                print(f"Processed and saved: {cleaned_img_path}")
            except Exception as e:
                print(f"Error processing {item}: {e}")


if __name__ == "__main__":
    input_folder_path = "data/images"
    output_folder_path = "data/clean_images"
    clean_image_data(input_folder_path, output_folder_path)
