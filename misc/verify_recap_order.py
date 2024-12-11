import os
import re
nba_team_names = ['CELTICS', 'MAVERICKS', 'NUGGETS', 'MAGIC', 'KNICKS', 'ROCKETS', 'PISTONS', 'JAZZ', 'WIZARDS',
                  'WARRIORS', 'SPURS', 'HAWKS', 'RAPTORS', 'GRIZZLIES', 'SUNS', 'PACERS', 'CLIPPERS',
                  'THUNDER', 'KINGS', 'BLAZERS', 'LAKERS', 'HEAT', '76ERS', 'BULLS', 'BUCKS', 'CAVALIERS', 'PELICANS',
                  'TIMBERWOLVES', 'HORNETS', 'NETS']

nba_team_abs = ['bos','dal','den','orl','nyk','hou','det','uta','was','gsw','sas','atl','tor','mem','phx','ind',
                  'lac','okc','sac','por','lal','mia','phi','chi','mil','cle','nop','min','cha','bkn']

def alphanum_key(s):
    """
    Helper function to extract numbers and text for natural sorting.
    """
    return [int(text) if text.isdigit() else float(text) if re.match(r'^[0-9]+\.[0-9]+$', text) else text.lower() for text in re.split('(\d+\.\d+|\d+)', s)]


def list_files_in_directory(directory):
    try:
        # Get the list of files and directories
        entries = os.listdir(directory)

        # Filter out only files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry)) and not
        entry.startswith('.')]
        files.sort(key=alphanum_key)
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    # Specify the directory path
    directory_path = '/Users/galishai/Dropbox/nba recaps/nba_com'

    # List the files in the directory
    file_names = list_files_in_directory(directory_path)
    count = 1
    with open('/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/full season data/rs_nbadotcom_links.txt', mode='r') as link_file:
        for line, file_name in zip(link_file, file_names):
            print("count: "+ str(count))
            found_line = False
            found_file = False
            team1 = ''
            team2 = ''
            for team_1 in nba_team_names:
                if team_1.lower() in file_name.lower():
                    team1 = team_1
                    break
            for team_2 in nba_team_names:
                if team_2 == team_1:
                    continue
                if team_2.lower() in file_name.lower():
                    team2 = team_2
                    break
            if team1.lower() in file_name.lower() and team2.lower() in file_name.lower():
                found_file = True
                team1_index = nba_team_names.index(team_1)
                team2_index = nba_team_names.index(team_2)
                team1_ab = nba_team_abs[team1_index]
                team2_ab = nba_team_abs[team2_index]
                assert team1_ab in line and team2_ab in line, "team1: "+team_1+"\nteam2: "+team_2+"\nteam1ab :"+team1_ab+"\nteam2ab: "+team2_ab + "\nnba.com_link: "+line+"\nrecap_name: "+file_name
                found_line = True
            assert found_line and found_file, "line: "+ line + "\nfile name: " + file_name
            count+=1
    print("OK!")
