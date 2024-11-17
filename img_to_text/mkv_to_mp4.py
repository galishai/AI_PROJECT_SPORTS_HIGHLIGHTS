import os
import ffmpeg

def convert_mkv_to_mp4(input_dir, output_dir=None):
    """
    Converts all MKV files in the input directory to MP4 format.

    :param input_dir: Path to the directory containing MKV files.
    :param output_dir: Path to save the converted MP4 files (default: same as input_dir).
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    if output_dir is None:
        output_dir = input_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.mkv'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.rsplit('.', 1)[0] + '.mp4')

            try:
                print(f"Converting {input_path} to {output_path}...")
                (
                    ffmpeg
                    .input(input_path)
                    .output(output_path, vcodec='copy', acodec='copy')
                    .run(overwrite_output=True)
                )
                print(f"Successfully converted: {output_path}")
            except ffmpeg.Error as e:
                print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    input_directory = input("Enter the path to the directory with MKV files: ").strip()
    output_directory = input("Enter the path to save MP4 files (leave blank to use the same directory): ").strip()
    output_directory = output_directory if output_directory else None

    convert_mkv_to_mp4(input_directory, output_directory)
