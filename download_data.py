import gdown

file_id = "1uBPbt79gABdUUmATIqO5jqrMzkTLfDTu"
url = f"https://drive.google.com/uc?id={file_id}"

output_path = "song_data_new.csv"

gdown.download(url, output_path, quiet=False)

