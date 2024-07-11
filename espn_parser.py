from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import csv
import re

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")

#returns player that assisted, or "None" if unassisted
def removeAssister(play_info):
    assist_pattern = re.compile(rf'\((.*?)\s+assists?\)')
    assist_match = assist_pattern.findall(play_info)
    if not assist_match:
        return ["None"]
    else:
        return assist_match

def get_roster(page_url):
    driver = webdriver.Chrome(options=chrome_options)

    # Open the page URL
    driver.get(page_url)

    # Wait for the content to load
    driver.implicitly_wait(10)  # Adjust the sleep time as needed
    # Get the page source after the dynamic content is loaded
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    dropdown_selector_home = ('#fittPageContainer > div:nth-child(2) > div > div:nth-child(6) > div > div > section.'
                              'Card.ShotChart > div > div > div.ShotChartControls.ml4 > div.ShotChartControls__wrap.pt5'
                              ' > div.ShotChartControls__team.ShotChartControls__team--home.w-100 > div.dropdown.dropd'
                              'own--sm.ShotChartControls__Dropdown.mr0.ml0.w-100 > select')
    dropdown_home = soup.select_one(dropdown_selector_home)
    dropdown_selector_away = ('#fittPageContainer > div:nth-child(2) > div > div:nth-child(6) > div > div > section.'
                              'Card.ShotChart > div > div > div.ShotChartControls.ml4 > div.ShotChartControls__wrap.pt5'
                              ' > div.ShotChartControls__team.ShotChartControls__team--away.w-100 > div.dropdown.drop'
                              'down--sm.ShotChartControls__Dropdown.mr0.ml0.w-100 > select')
    dropdown_away = soup.select_one(dropdown_selector_away)
    if dropdown_home:
        player_names_home = [option['value'] for option in dropdown_home.find_all('option')]
    if dropdown_away:
        player_names_away = [option['value'] for option in dropdown_away.find_all('option')]
    #players_text = [team.get_text(strip=True) for team in teams]
    del player_names_home[0]
    del player_names_away[0]
    driver.quit()
    return player_names_home, player_names_away




def get_play_component_data(page_url, stage, game_num):
    # Initialize WebDriver (make sure to specify the path to your WebDriver)

    home_roster, away_roster = get_roster(page_url)
    driver = webdriver.Chrome(options=chrome_options)

    # Open the page URL
    driver.get(page_url)

    # Wait for the content to load
    driver.implicitly_wait(10)  # Adjust the sleep time as needed
    # Get the page source after the dynamic content is loaded
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    teams = soup.find_all(True, class_=['fw-medium n7 ml2'])
    teams_text = [team.get_text(strip=True) for team in teams]
    records = soup.find_all(True, class_=['Gamestrip__Record db clr-gray-03 n9'])
    records_text = [record.get_text(strip=True) for record in records]
    away_wins = int(records_text[0].split('-')[0])
    away_losses = int(records_text[0].split(",")[0].split('-')[1])
    away_win_percentage = round(away_wins/(away_wins + away_losses) * 100, 2)
    home_wins = int(records_text[1].split('-')[0])
    home_losses = int(records_text[1].split(",")[0].split('-')[1])
    home_win_percentage = round(home_wins/(home_wins + home_losses) * 100, 2)

    win_diff = abs(int(home_wins) - int(away_wins))

    home_team = teams_text[1]
    away_team = teams_text[0]

    play_components_text = []
    quarter_texts = ['1st', '2nd', '3rd', '4th', 'OT', '2 OT']


    driver.execute_script("window.scrollBy(0, 200);")
    for quarter_text in quarter_texts:
        # Click on the quarter button
        existence_check = driver.find_elements(By.XPATH,
                                             f'//button[contains(@class, "Button--unstyled tabs__link") and text()'
                                             f'="{quarter_text}"]')
        if not existence_check:
            break
        quarter_button = existence_check[0]
        quarter_button.click()
        driver.implicitly_wait(5)  # Adjust the sleep time as needed
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        play_times_q = soup.find_all(True, class_=['playByPlay__time Table__TD'])
        play_info_q = soup.find_all(True,
                                  class_=['playByPlay__text tl Table__TD', 'playByPlay__text tl clr-btn Table__TD'])
        if not play_times_q:
            print("Parsing Error 1!" + quarter_text)
            return -1
        if not play_info_q:
            print("Parsing Error 2!" + quarter_text)
            return -1
        '''if len(play_times) != len(play_info):
            print("Parsing Error 3!")
            return -1
            '''

        play_times_text = [play_time.get_text(strip=True) for play_time in play_times_q]
        play_info_text = [info.get_text(strip=True) for info in play_info_q]
        for play_time, play_info in zip(play_times_text, play_info_text):
            #print(play_info)
            if "vs" in play_info:
                continue
            splitted = play_info.split()
            player_name = "cannot identify"
            if (home_team in play_info.upper() or away_team in play_info.upper() or "enters" in play_info or "End of" in\
                    play_info or "Start of" in play_info or "delay" in play_info or "timeout" in play_info or "Game"
                    in play_info):
                player_name = "None"
                continue
            elif "jr" in splitted[2].lower() or "II" in splitted[2].lower() or "III" in splitted[2].lower():
                player_name = splitted[0] + " " + splitted[1] + " " + splitted[2]
                play_info = ' '.join(play_info.split()[3:])
            else:
                player_name = splitted[0] + " " + splitted[1]
                play_info = ' '.join(play_info.split()[2:])

            assister = removeAssister(play_info)
            play_info = re.sub(r'\(.*?\)', '', play_info).strip()
            total_games = min(home_wins + home_losses, away_wins + away_losses)
            if player_name in home_roster:
                win_percentage = home_win_percentage
            elif player_name in away_roster:
                win_percentage = away_win_percentage
            else:
                win_percentage = None
            play_components_text.append([play_time, play_info, quarter_text, home_team, away_team, player_name,
                                         assister[0], stage, game_num, win_diff, total_games, win_percentage])
    if not play_components_text:
        print("No play components found. Verify the class name or structure of the HTML.")
    else:
        print(f"Found {len(play_components_text)} plays.")
    driver.quit()
    return play_components_text

def main():
    page_url = input("Enter AUTO to generate default. Or enter ESPN game urls manually and enter END to finish the "
                     "program")
    play_data = [["Time", "Play", "Quarter", "Home Team", "Away Team", "Name", "Assister", "Stage", "Game Num", "Win Difference(Abs)", "Games Played", "Win percentage"]]
    if page_url == "AUTO":
        with open('game_urls.txt', 'r') as file:
            for line in file:
                # Strip the newline character and any leading/trailing whitespace
                cleaned_line = line.strip()
                split_line = cleaned_line.split()
                stage = split_line[0]
                game_num = split_line[1]
                page_url = split_line[2]
                play_data += get_play_component_data(page_url, stage, game_num)

    else:
        while page_url != "END":
            stage = input("enter stage")
            game_num = input("enter game num")
            play_data += get_play_component_data(page_url, stage, game_num)
            if play_data == -1:
                print("FATAL ERROR")
                return
            #save_to_file(play_data, 'play_components.txt')
            page_url = input()
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(play_data)
    print(f"Play component text saved to output.csv")
    print("OK :)")

if __name__ == '__main__':
    main()

