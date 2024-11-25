import re

# Path to the file
input_file = 'cmds.txt'
output_file = 'ordered_cmds_v2.txt'

# Read the content of the file
with open(input_file, 'r') as file:
    lines = file.readlines()

renumbered_lines = []
video_number = 1

# Process each line
for line in lines:
    # Replace "terminal/mkvmerge.sh" with the full path
    line = line.replace("terminal/mkvmerge.sh",
                        "/Users/galishai/Downloads/widefrog_v2_10_0_python_source_code/terminal/mkvmerge.sh")

    # Extract relevant information from the game recap (teams and scores)
    match = re.search(r'Game_Recap_(.*?)_\d+_(.*?)_\d+', line)
    if match:
        team1 = match.group(1)
        team2 = match.group(2)
        new_name = f"Game_Recap_{video_number:02d}_{team1}_vs_{team2}"

        # Replace Video_X directory and game recap folder with the new informative name
        line = re.sub(r'Video_\d+', f'Video_{video_number:02d}_{team1}_vs_{team2}', line)
        line = re.sub(r'Game_Recap_\w+_\d+_\w+_\d+', new_name, line)

    renumbered_lines.append(line)
    video_number += 1

# Write the renumbered commands to a new file
with open(output_file, 'w') as file:
    file.writelines(renumbered_lines)

print(f"Renumbered commands with informative names and updated mkvmerge path saved to {output_file}")
