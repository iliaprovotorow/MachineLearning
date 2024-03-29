import csv
import math
from graphviz import Digraph    # модуль визуализации алгоритма

class node:
    question = None     #
    left = None         # задаёт атрибуты для вопроса
    right = None        #

    def __init__(self):
        pass

def read_csv(filename: str, out_column: str):   # возвращает словарь нормализованных данных, где ключи словаря - имена столбцов,
    data = []                                   # а значения - данные из соответствующих столбцов
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Находим минимальные и максимальные значения для каждого признака
    min_val = {col: min(float(row[col]) for row in data) for col in data[0] if col != out_column}
    max_val = {col: max(float(row[col]) for row in data) for col in data[0] if col != out_column}

    # Нормализация данных
    for item in data:
        for key in item:
            if key != out_column:
                item[key] = (float(item[key]) - min_val[key]) / (max_val[key] - min_val[key])

    return data

def get_column(data, column: str):  # извлекает определённый столбец из списка словарей
    out = []
    for item in data:
        out.append(item[column])
    return out

def entropy(items: list) -> float:  # вычисляет энтропию списка элементов
    size = len(items)
    unique = []
    counts = []
    for item in items:
        if item not in unique:
            unique.append(item)
            counts.append(items.count(item))
    out = 0
    for item in counts:
        out += item / size * math.log2(item / size)
    return float(-out)


# Вычисляет информационный выигрыш с учетом исходной энтропии и энтропии левой и правой частей
def infogain(source_entropy: float,
             left_entropy: float,
             right_entropy: float,
             left_count: int,
             right_count: int) -> float:
    size = left_count + right_count
    return float(source_entropy - left_count / size * left_entropy - right_count / size * right_entropy)

def split(data, field: str, value: float):      # Разделяет набор данных на две части на основе заданного поля и значения
    first = []
    second = []
    for item in data:
        if item[field] <= value:
            first.append(item)
        else:
            second.append(item)
    return [first, second]


# Вычисляет информационный выигрыш для конкретного вопроса
def perform_question(data, out_column: str, source_entropy: float, question) -> float:
    values = split(data, question[0], question[1])
    left = get_column(values[0], out_column)
    right = get_column(values[1], out_column)
    return infogain(source_entropy, entropy(left), entropy(right), len(left), len(right))


# Находит лучший вопрос для разделения данных с максимальным информационным выигрышем
def find_best_question(data, out_column: str):
    best_question = ('', 0)
    max_ig = 0
    source_entropy = entropy(get_column(data, out_column))
    for col in data[0]:
        if col == out_column:
            continue
        questions = set(get_column(data, col))
        i = 0
        for q in questions:
            ig = perform_question(data, out_column, source_entropy, (col, q))
            if (ig > max_ig):
                max_ig = ig
                best_question = (col, q)
    if max_ig == 0:
        return None
    return best_question


# Рекурсивное построение дерева принятия решений
def train_recursive(data, out_column: str, _node: node, depth: int, graph=None, parent=None, branch_label=None):
    question = find_best_question(data, out_column)
    if question == None:
        if len(data) != 0:
            _node.question = str(data[0][out_column])
        return
    _node.question = question
    _node.left = node()
    _node.right = node()
    next = split(data, question[0], question[1])
    if graph is not None:
        if parent is not None:
            parent_id = str(id(parent))
            left_child_id = str(id(next[0]))
            right_child_id = str(id(next[1]))
            graph.node(left_child_id, str(next[0][0]))
            graph.node(right_child_id, str(next[1][0]))
            graph.edge(parent_id, left_child_id, label=str(branch_label))
            graph.edge(parent_id, right_child_id, label=str(branch_label))
        train_recursive(next[0], out_column, _node.left, depth + 1, graph, next[0], question[0] + " <= " + str(question[1]))
        train_recursive(next[1], out_column, _node.right, depth + 1, graph, next[1], question[0] + " > " + str(question[1]))
    else:
        train_recursive(next[0], out_column, _node.left, depth + 1)
        train_recursive(next[1], out_column, _node.right, depth + 1)


# Обучает классификатор на основе дерева принятия решений с использованием обучающих данных
def train(data, out_column: str) -> node:
    root = node()
    graph = Digraph(comment='Decision Tree')

    # Находим минимальные и максимальные значения для каждого признака
    min_val = {col: min(float(row[col]) for row in data) for col in data[0] if col != out_column}
    max_val = {col: max(float(row[col]) for row in data) for col in data[0] if col != out_column}

    train_recursive(data, out_column, root, 0, graph)
    graph.render('D:/PycharmProjects/MachineLearning/decision_tree', format='png')
    return root


# Рекурсивно классифицирует объект с помощью обученного дерева принятия решений
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


# Классифицирует один объект с использованием обученного дерева принятия решений
def classify(obj, model: node) -> str:
    result = ['']
    classify_recursive(obj, model, result)
    return result[0]


# Основная функция, где выполняется обучение и тестирование классификатора на основе дерева принятия решений
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
    print(f"Accuracy:\t{(total - misses) / total}")

if __name__ == '__main__':
    main()
