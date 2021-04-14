import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('input.png')
plt.imshow(image)

thickness = 0


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


# Все пиксели = 0, полная непрозрачность
def is_black(rgb: tuple) -> bool:
    return rgb[0] == 0.0 and rgb[1] == 0.0 and rgb[2] == 0.0 and rgb[3] == 1.0


# Все элементы в котреже равны
def are_same(items: tuple) -> bool:
    return all(x == items[0] for x in items)


def find_vertical_lines(image: np.ndarray) -> tuple:
    global thickness
    first_start = None
    second_start = None
    x = 0
    y = 0
    while y < image.shape[0]:
        while x < image.shape[1]:
            if is_black(image[y, x]):
                if first_start is None:
                    first_start = Point(x, y)
                else:
                    second_start = Point(x, y)
                    y = image.shape[0]
                    break
                thickness = 0
                while is_black(image[y, x]):
                    x += 1
                    thickness += 1
            x += 1
        x = 0
        y += 1

    first_end = None
    second_end = None
    y = image.shape[0] - 1
    x = 0
    while y >= 0:
        while x < image.shape[1]:
            if is_black(image[y, x]):
                if first_end is None:
                    first_end = Point(x, y)
                else:
                    second_end = Point(x, y)
                    return (first_start, first_end), (second_start, second_end)
                while is_black(image[y, x]):
                    x += 1
            x += 1
        x = 0
        y -= 1


def find_horizontal_lines(image: np.ndarray) -> tuple:
    first_start = None
    second_start = None
    x = 0
    y = 0
    while x < image.shape[1]:
        while y < image.shape[0]:
            if is_black(image[y, x]):
                if first_start is None:
                    first_start = Point(x, y)
                else:
                    second_start = Point(x, y)
                    x = image.shape[1]
                    break
                while is_black(image[y, x]):
                    y += 1
            y += 1
        y = 0
        x += 1

    first_end = None
    second_end = None
    x = image.shape[1] - 1
    y = 0
    while x >= 0:
        while y < image.shape[0]:
            if is_black(image[y, x]):
                if first_end is None:
                    first_end = Point(x, y)
                else:
                    second_end = Point(x, y)
                    return (first_start, first_end), (second_start, second_end)
                while is_black(image[y, x]):
                    y += 1
            y += 1
        y = 0
        x -= 1


def get_center(rectangle: tuple) -> tuple:
    return (rectangle[0].x + rectangle[1].x)//2, \
           (rectangle[0].y + rectangle[1].y)//2


# Пустой сектор
def is_empty(image: np.ndarray, rectangle: tuple) -> bool:
    x, _ = get_center(rectangle)
    for y in range(rectangle[0].y + 1, rectangle[1].y):
        if is_black(image[y, x]):
            return False
    return True


def is_circle(image: np.ndarray, rectangle: tuple) -> bool:
    x, y = get_center(rectangle)
    return not is_black(image[y, x])


def is_cross(image: np.ndarray, rectangle: tuple) -> bool:
    x, y = get_center(rectangle)
    return is_black(image[y, x])


def recognize_shape(image: np.ndarray, rectangle: tuple) -> str or None:
    if is_empty(image, rectangle):
        return ''
    if is_circle(image, rectangle):
        return 'O'
    if is_cross(image, rectangle):
        return 'X'
    return None  # undefined


vert_line1, vert_line2 = find_vertical_lines(image)
hor_line1, hor_line2 = find_horizontal_lines(image)

field = np.empty(shape=(3, 3), dtype=object)

# Прямоугольник (сектор) однозначно задается двумя диагональными точками
# Конструируем игровое поле
field[0, 0] = recognize_shape(image, (
    Point(vert_line1[0].x, vert_line1[0].y + thickness - 1),
    Point(hor_line1[0].x + thickness - 1, hor_line1[0].y)
))

field[0, 1] = recognize_shape(image, (
    Point(vert_line2[0].x, vert_line2[0].y + thickness - 1),
    Point(vert_line1[0].x + thickness - 1, hor_line1[0].y)
))

field[0, 2] = recognize_shape(image, (
    Point(hor_line1[1].x - thickness + 1, vert_line2[0].y + thickness - 1),
    Point(vert_line2[0].x + thickness - 1, hor_line1[0].y)
))

field[1, 0] = recognize_shape(image, (
    Point(vert_line1[0].x, hor_line1[0].y + thickness - 1),
    Point(hor_line2[0].x + thickness - 1, hor_line2[0].y)
))

field[1, 1] = recognize_shape(image, (
    Point(vert_line2[0].x, hor_line1[0].y + thickness - 1),
    Point(vert_line1[0].x + thickness, hor_line2[0].y)
))

field[1, 2] = recognize_shape(image, (
    Point(hor_line1[1].x - thickness + 1, hor_line1[1].y + thickness - 1),
    Point(vert_line2[0].x + thickness - 1, hor_line2[1].y)
))

field[2, 0] = recognize_shape(image, (
    Point(vert_line1[0].x, hor_line2[0].y + thickness - 1),
    Point(hor_line2[0].x + thickness - 1, vert_line1[1].y - thickness + 1)
))

field[2, 1] = recognize_shape(image, (
    Point(vert_line2[1].x, hor_line2[1].y + thickness - 1),
    Point(vert_line1[1].x + thickness - 1, vert_line1[1].y - thickness + 1)
))

field[2, 2] = recognize_shape(image, (
    Point(hor_line2[1].x - thickness + 1, hor_line2[1].y),
    Point(vert_line2[1].x + thickness - 1, vert_line2[1].y - thickness + 1)
))


# Ищем победителя
if are_same(field[0]):
    _, y = get_center((Point(hor_line1[1].x, vert_line2[0].y),
                       Point(vert_line2[0].x, hor_line1[0].y)))
    plt.plot([hor_line1[0].x, hor_line1[1].x], y)
elif are_same(field[1]):
    _, y = get_center((Point(hor_line1[1].x, hor_line1[1].y),
                       Point(vert_line2[0].x, hor_line2[1].y)))
    plt.plot([hor_line1[0].x, hor_line1[1].x], y)
elif are_same(field[2]):
    _, y = get_center((Point(hor_line2[1].x, hor_line2[1].y),
                       Point(vert_line2[1].x, vert_line2[1].y)))
    plt.plot([hor_line1[0].x, hor_line1[1].x], y)
elif are_same(field[:, 0]):
    x, _ = get_center((Point(vert_line1[0].x, vert_line1[0].y),
                       Point(hor_line1[0].x, hor_line1[0].y)))
    plt.plot([x, x], [vert_line1[0].y, vert_line1[1].y])
elif are_same(field[:, 1]):
    x, _ = get_center((Point(vert_line2[0].x, vert_line2[0].y),
                       Point(vert_line1[0].x, hor_line1[0].y)))
    plt.plot([x, x], [vert_line1[0].y, vert_line1[1].y])
elif are_same(field[:, 2]):
    x, _ = get_center((Point(hor_line1[1].x, vert_line1[0].y),
                       Point(vert_line2[0].x, hor_line1[0].y)))
    plt.plot([x, x], [vert_line1[0].y, vert_line1[1].y])
elif are_same((field[0, 0], field[1, 1], field[2, 2])):
    plt.plot([hor_line1[0].x, hor_line1[1].x],
             [vert_line1[0].y, vert_line1[1].y])
elif are_same((field[0, 2], field[1, 1], field[2, 0])):
    plt.plot([hor_line1[1].x, hor_line1[0].x],
             [vert_line1[0].y, vert_line1[1].y])
else:
    print('No winners')

plt.savefig("result.png")

plt.show()
