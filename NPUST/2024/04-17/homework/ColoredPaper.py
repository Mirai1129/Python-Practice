def count_total_area(papers):
    total_area = 0
    for paper in papers:
        quantity = paper['quantity']
        shape = paper['shape']
        side_length = paper['side_length']
        if shape == 'square':
            area = quantity * side_length ** 2
        elif shape == 'triangle':
            area = quantity * side_length ** 2 * 0.433
        elif shape == 'circle':
            area = quantity * (side_length / 4) * 3.14
        else:
            area = 0
        total_area += area
    return total_area


def get_paper_info():
    papers = []
    while True:
        shape = input("請輸入色紙的形狀 (square/triangle/circle)，輸入 'done' 結束程式: ")
        if shape == 'done':
            break
        quantity = int(input("請輸入數量: "))
        side_length = float(input("請輸入邊長或半徑: "))
        papers.append({'quantity': quantity, 'shape': shape, 'side_length': side_length})
    return papers


def main():
    papers = get_paper_info()
    total_area = count_total_area(papers)
    print(f"整包色紙的總面積: {total_area}")


if __name__ == '__main__':
    main()
