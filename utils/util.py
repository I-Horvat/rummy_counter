import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

symbols = ['2♥', '2♦', '2♣', '2♠', '3♥', '3♦', '3♣', '3♠', '4♥', '4♦', '4♣', '4♠', '5♥', '5♦', '5♣', '5♠', '6♥', '6♦',
           '6♣', '6♠', '7♥', '7♦', '7♣', '7♠', '8♥', '8♦', '8♣', '8♠', '9♥', '9♦', '9♣', '9♠', '10♥', '10♦', '10♣',
           '10♠', 'J♥', 'J♦', 'J♣', 'J♠', 'Q♥', 'Q♦', 'Q♣', 'Q♠', 'K♥', 'K♦', 'K♣', 'K♠', 'A♥', 'A♦', 'A♣', 'A♠',
           'JOKER']

def symbol_to_int(symbol):
    if symbol in symbols:
        return [symbols.index(symbol) + 1]
    else:
        return [0]

symbol_to_point= {'2♥':2,'2♦':2,'2♣':2,'2♠':2,'3♥':3,'3♦':3,'3♣':3,'3♠':3,'4♥':4,'4♦':4,'4♣':4,'4♠':4,'5♥':5,'5♦':5,'5♣':5,'5♠':5,
                  '6♥':6,'6♦':6,'6♣':6,'6♠':6,'7♥':7,'7♦':7,'7♣':7,'7♠':7,'8♥':8,'8♦':8,'8♣':8,'8♠':8,'9♥':9,'9♦':9,'9♣':9,'9♠':9,
                  '10♥':10,'10♦':10,'10♣':10,'10♠':10,'J♥':10,'J♦':10,'J♣':10,'J♠':10,'Q♥':10,'Q♦':10,'Q♣':10,'Q♠':10,'K♥':10,'K♦':10,'K♣':10,'K♠':10,
                    'A♥':10,'A♦':10,'A♣':10,'A♠':10,'JOKER':20}
def check_bbbox_integrity(bbox, image, min_area=12000):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[-1], x2)
    y2 = min(image.shape[-2], y2)

    if x2 > x1 and y2 > y1:
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            return [x1, y1, x2, y2]
    return None


def write_to_log(file, message):
    try:
        with open(file, 'a') as f:
            f.write(message + "\n")
    except FileNotFoundError:
        print(f"Error: The file {file} was not found.")
    except IOError:
        print("IO error occurred when writing to the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def validate_model(model, val_loader, device):
    model.train()
    total_val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
    return total_val_loss / len(val_loader)


def early_stopping(val_loss, best_val_loss, patience_counter, patience_limit=5):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        return False, best_val_loss, patience_counter
    else:
        patience_counter += 1
        if patience_counter > patience_limit:
            return True, best_val_loss, patience_counter
    return False, best_val_loss, patience_counter


def train_one_epoch(model, train_loader, optimizer, log_file_name):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        if torch.isnan(losses):
            write_to_log(log_file_name, f"NaN detected in batch")
            continue

        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    avg_loss = total_loss / len(train_loader)
    write_to_log(log_file_name, f"Total loss: {total_loss}")
    write_to_log(log_file_name, f"Average loss: {avg_loss}")

    return avg_loss
