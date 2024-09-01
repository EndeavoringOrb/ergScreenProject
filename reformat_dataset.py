import pandas as pd

def convert_seconds_to_string(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if int(hours) > 0:
        time_string = f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
    else:
        time_string = f"{int(minutes)}:{int(seconds):02}"
    return time_string

# Function to get text based on input parameters
def get_text(rows):
    texts = []
    for index, row in rows.iterrows():
        row_num = row['row_num']
        time = row['time']
        meters = row['meters']
        avg_split = row['avg_split']
        avg_spm = row['avg_spm']
        text = f"{convert_seconds_to_string(time)}, {meters}, {convert_seconds_to_string(avg_split)}, {avg_spm}"
        texts.append(text)
    return '|'.join(texts)

# Read the CSV file
df = pd.read_csv('dataset.csv')

# Group rows by image_path and apply the get_text function to each group
grouped_data = df.groupby('image_path').apply(lambda x: get_text(x))

# Create a new DataFrame with image_path and formatted text
formatted_data = pd.DataFrame({'path': grouped_data.index, 'text': grouped_data.values})

# Save the reformatted data into a new CSV file
formatted_data.to_csv('dataset_path_text.csv', index=False)