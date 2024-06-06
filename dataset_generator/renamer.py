import os

def rename_images(image_folder):
    suits = {
        'S': '♠',
        'H': '♥',
        'D': '♦',
        'C': '♣'
    }
    values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    image_files = os.listdir(image_folder)
    for image_file in image_files:
        name, ext = os.path.splitext(image_file)

        suit_code = name[-1]
        value = name[:-1]

        if suit_code in suits and value in values:
            suit_symbol = suits[suit_code]
            new_name = f"{value}{suit_code}{suit_symbol}{ext}"
            os.rename(os.path.join(image_folder, image_file), os.path.join(image_folder, new_name))
            print(f"Renamed {image_file} to {new_name}")

if __name__ == '__main__':
    image_folder = 'images/cropped_previous'
    rename_images(image_folder)
    print("Images renamed successfully")
