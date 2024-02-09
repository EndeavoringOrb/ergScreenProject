import csv
import numpy as np

def to_one_hot(num, size=11):
    return [1 if i == num else 0 for i in range(-1, size-1)]

def row_to_classes(row):
    image_path, row_num, time, meters, avg_split, avg_spm = row
    row_num = int(row_num)
    time = float(time)
    meters = float(meters)
    avg_split = float(avg_split)
    avg_spm = float(avg_spm)
    # 17x11
    # 1        2      3    4      5    6
    # time hr, 10min, min, 10sec, sec, dec
    time_hr = time // 3600
    minutes = (time % 3600) // 60
    minutes_10 = minutes // 10
    minutes = (minutes % 10) // 1
    seconds = time % 60
    seconds_10 = seconds // 10
    decimal_seconds = ((seconds % 1) * 10) // 1
    seconds = (seconds % 10) // 1
    if time < 3600:
        time_hr = -1
    if time < 600:
        minutes_10 = -1
    if time < 60:
        minutes = -1
    if time < 10:
        seconds_10 = -1
    if time < 1:
        seconds = -1
    #        7      8     9    10  11
    # meter: 10000, 1000, 100, 10, 1
    meter_10000 = meters // 10_000
    meter_1000 = (meters % 10_000) // 1000
    meter_100 = (meters % 1000) // 100
    meter_10 = (meters % 100) // 10
    meter_1 = (meters % 10) // 1
    if meters < 10_000:
        meter_10000 = -1
    if meters < 1000:
        meter_1000 = -1
    if meters < 100:
        meter_100 = -1
    if meters < 10:
        meter_10 = -1
    #       12   13     14   15
    # split min, 10sec, sec, dec
    split_minutes = (avg_split % 3600) // 60
    split_seconds = avg_split % 60
    split_seconds_10 = split_seconds // 10
    split_decimal_seconds = ((split_seconds % 1) * 10) // 1
    split_seconds = (split_seconds % 10) // 1
    if avg_split < 60:
        split_minutes = -1
    #      16  17
    # spm: 10, 1
    spm_10 = avg_spm // 10
    spm = avg_spm % 10
    return_x_array = [time_hr, minutes_10, minutes, seconds_10, seconds, decimal_seconds, meter_10000, meter_1000, meter_100, meter_10, meter_1, split_minutes, split_seconds_10, split_seconds, split_decimal_seconds, spm_10, spm]
    return row_num, return_x_array

def get_row_sizes():
    classes = []
    for i in range(9):
        classes.append([])
    with open("dataset.csv", mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)  # Read all rows and convert to a list
        rows = rows[:int(len(rows)*0.8)]
        for row in rows:
            row_num, x_arr = row_to_classes(row)
            classes[row_num].append(x_arr)
    classes = [np.asarray(i) for i in classes]
    for i in range(len(classes)):
        print(f"# of Row {i+1} Images: {classes[i].shape[0]}")

def get_row_class_sizes(test_row_num):
    class_sizes = np.zeros((17, 11))
    with open("dataset.csv", mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = list(reader)  # Read all rows and convert to a list
        rows = rows[:int(len(rows)*0.8)]
        for row in rows:
            row_num, x_arr = row_to_classes(row)
            x_arr = [to_one_hot(i) for i in x_arr]
            x_arr = np.array(x_arr)
            if test_row_num == row_num:
                class_sizes += x_arr
    names = ["time_hr", "minutes_10", "minutes", "seconds_10", "seconds", "decimal_seconds", "meter_10000", "meter_1000", "meter_100", "meter_10", "meter_1", "split_minutes", "split_seconds_10", "split_seconds", "split_decimal_seconds", "spm_10", "spm"]
    for i in range(len(names)):
        print(f"{names[i]:22}: {class_sizes[i]}")

if __name__ == "__main__":
    get_row_sizes()
    get_row_class_sizes(0)
    #for i in range(9):
    #    get_row_class_sizes(i)
