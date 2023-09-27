def clear_lines(num_lines):
    for _ in range(num_lines):
        print("\033[F\033[K", end="")
        #sys.stdout.write("\033[F")  # Move cursor up
        #sys.stdout.write("\033[K")  # Clear the line

def move_up_lines(num_lines):
    for _ in range(num_lines):
        print("\033[F", end="")

def str_to_float_time(time_string):
    return_time = 0
    times = time_string.strip().split(" ")
    if len(times) == 3:
        return_time += 3600 * int(times[0])
        return_time += 60 * int(times[1])
        return_time += float(times[2])
    elif len(times) == 2:
        return_time += 60 * int(times[0])
        return_time += float(times[1])
    elif len(times) == 1:
        return_time += float(times[0])
    return return_time