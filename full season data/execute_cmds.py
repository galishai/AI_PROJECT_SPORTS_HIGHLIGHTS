import subprocess

# Path to the input file containing commands
input_file = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/full season data/ordered_cmds_v2.txt'  # Replace with the path to your file

STARTING_COUNT = 55
# Open and read all lines from the file
with open(input_file, 'r') as file:
    commands = file.readlines()

# Loop through and execute each command
for idx, command in enumerate(commands, start=STARTING_COUNT):
    command = command.strip()  # Remove leading/trailing whitespace

    # Ignore empty lines and lines starting with '#'
    if not command or command.startswith('#'):
        continue

    print(f"Executing command {idx}: {command}")

    try:
        # Execute the command
        result = subprocess.run(command, shell=True, text=True, capture_output=True)

        # Log the command's output
        print(f"Command {idx} Output:\n{result.stdout}")

        # Log any errors
        if result.stderr:
            print(f"Command {idx} Errors:\n{result.stderr}")

    except Exception as e:
        print(f"An error occurred while executing command {idx}: {e}")
