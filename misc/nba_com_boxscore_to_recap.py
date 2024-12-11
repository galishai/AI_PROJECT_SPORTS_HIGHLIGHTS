# Define input and output file paths
input_file = "../full season data/rs_nbadotcom_links.txt"  # Replace with the name of your input file
output_file = "../full season data/game_recap_links.txt"  # Name of the output file

# Open the input file and process each line
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Strip whitespace (like newlines) and append the query parameter
        updated_link = line.strip() + "?watchRecap=true"
        # Write the updated link to the output file
        outfile.write(updated_link + "\n")

print(f"Updated links saved to {output_file}")
