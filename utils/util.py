symbols = ['2♥', '2♦', '2♣', '2♠', '3♥', '3♦', '3♣', '3♠', '4♥', '4♦', '4♣', '4♠', '5♥', '5♦', '5♣', '5♠', '6♥', '6♦',
           '6♣', '6♠', '7♥', '7♦', '7♣', '7♠', '8♥', '8♦', '8♣', '8♠', '9♥', '9♦', '9♣', '9♠', '10♥', '10♦', '10♣',
           '10♠', 'J♥', 'J♦', 'J♣', 'J♠', 'Q♥', 'Q♦', 'Q♣', 'Q♠', 'K♥', 'K♦', 'K♣', 'K♠', 'A♥', 'A♦', 'A♣', 'A♠',
           'JOKER']

def symbol_to_int(symbol):
    if symbol in symbols:
        return [symbols.index(symbol) + 1]
    else:
        return [0]

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
