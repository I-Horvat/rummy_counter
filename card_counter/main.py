import json
import os


def count_cards_per_class(path):
    card_count = {}
    for folder in os.listdir(path):
        folder_path=os.path.join(path,folder,'regions.json')
        try:
            with open(folder_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for region in data:
                    card_count[region['tagName']] = card_count.get(region['tagName'], 0) + 1
        except:
            print(f"Error while reading {folder_path}. Skipping this folder.")
            continue
    return card_count

if __name__ == '__main__':
    # path='../images/total_generated_augmented'
    path='../data_augmentation/augmented_final'
    card_count=count_cards_per_class(path)
    print(card_count)
    print(card_count.keys().__len__())