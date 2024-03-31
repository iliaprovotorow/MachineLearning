import csv
import math

class node:
    question = None
    left = None
    right = None
    def __init__(self):
        pass

def read_csv(filename: str, out_column: str):
    data = [ ]
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    for item in data:
        for key in item:
            if key != out_column:
                item[key] = float(item[key])
    return data

def get_column(data, column: str):
    out = [ ]
    for item in data:
        out.append(item[column])
    return out

def entropy(items: list) -> float:
    size = len(items)
    unique = [ ]
    counts = [ ]
    for item in items:
        if item not in unique:
            unique.append(item)
            counts.append(items.count(item))
    out = 0
    for item in counts:
        out += item/size * math.log2(item/size)
    return float(-out)

def infogain(source_entropy: float,
             left_entropy: float,
             right_entropy: float,
             left_count: int,
             right_count: int) -> float:
    size = left_count + right_count
    return float( source_entropy - left_count/size * left_entropy - right_count/size * right_entropy )

def split(data, field: str, value: float):
    first = []
    second = []
    for item in data:
        if item[field] <= value:
            first.append(item)
        else:
            second.append(item)
    return [first, second]

def perform_question(data, out_column: str, source_entropy: float, question) -> float:
    values = split(data, question[0], question[1])
    left = get_column(values[0], out_column)
    right = get_column(values[1], out_column)
    return infogain(source_entropy, entropy(left), entropy(right), len(left), len(right))

def find_best_question(data, out_column: str):
    best_question = ( '', 0 )
    max_ig = 0
    source_entropy = entropy(get_column(data, out_column))
    for col in data[0]:
        if col == out_column:
            continue
        questions = set(get_column(data, col))
        i = 0
        for q in questions:
            ig = perform_question(data, out_column, source_entropy, (col, q) )
            if (ig > max_ig):
                max_ig = ig
                best_question = (col, q)
    if max_ig == 0:
        return None
    return best_question

def train_recursive(data, out_column: str, _node: node, depth: int):
    question = find_best_question(data, out_column)
    if question == None:
        if len(data) != 0:
            _node.question = str(data[0][out_column])
        return
    _node.question = question
    _node.left = node()
    _node.right = node()
    next = split(data, question[0], question[1])
    train_recursive(next[0], out_column, _node.left, depth + 1)
    train_recursive(next[1], out_column, _node.right, depth + 1)
    
def train(data, out_column: str) -> node:
    root = node()
    train_recursive(data, out_column, root, 0)
    return root

def classify_recursive(obj, model: node, result: str):
    if isinstance(model.question, str):
        result[0] = model.question
        return
    if model.question == None:
        return
    if obj[model.question[0]] <= model.question[1]:
        if model.left != None:
            classify_recursive(obj, model.left, result)
    else:
        if model.right != None:
            classify_recursive(obj, model.right, result)

def classify(obj, model: node) -> str:
    result = ['']
    classify_recursive(obj, model, result)
    return result[0]

def main():
    out_column = 'class'
    train_data = read_csv("train.csv", out_column)
    test_data = read_csv("test.csv", None)
    test_data_src = read_csv("test-src.csv", out_column)

    model = train(train_data, out_column)

    total = 0
    misses = 0
    for item in test_data:
        item[out_column] = classify(item, model)
        print(f"{str(item)} : {item == test_data_src[total]}")
        if item != test_data_src[total]:
            misses += 1
        total += 1
    
    print("--------------------------")
    print(f"Total:\t\t{total}")
    print(f"Correct:\t{total - misses}")
    print(f"Accuracy:\t{(total-misses)/total}")
    
    
if __name__ == '__main__':
    main()