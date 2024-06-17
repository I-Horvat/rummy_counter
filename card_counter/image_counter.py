import os


def count_folder_images(folder_path):
    count=0
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)):
            count+=1
    return count

if __name__ == '__main__':
    folder_path = '../images/cleaned_and_total'
    print(count_folder_images(folder_path))
    print("Done")