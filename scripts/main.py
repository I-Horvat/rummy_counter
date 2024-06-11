import shutil

def zip_folder(folder_path, output_path):
    shutil.make_archive(output_path, 'zip', folder_path)

def main(folder_to_zip,output_zip):

    zip_folder(folder_to_zip,output_zip)



if __name__ == '__main__':
    folder_to_zip='../images/total_generated_augmented'
    output_zip='zipped_everything'
    main(folder_to_zip,output_zip)